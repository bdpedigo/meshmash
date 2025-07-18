import time
from typing import NamedTuple

import numpy as np
import pandas as pd
from fast_simplification import replay_simplification, simplify

from .agglomerate import (
    agglomerate_mesh,
    agglomerate_split_mesh,
    aggregate_features,
    fix_split_labels_and_features,
)
from .decompose import compute_hks
from .graph import condense_mesh_to_graph
from .laplacian import compute_vertex_areas
from .split import MeshStitcher
from .types import interpret_mesh
from .utils import (
    compute_distances_to_point,
    expand_labels,
    threshold_mesh_by_component_size,
)


class Result(NamedTuple):
    simple_mesh: tuple
    mapping: np.ndarray
    stitcher: MeshStitcher
    simple_features: pd.DataFrame
    simple_labels: np.ndarray
    condensed_features: pd.DataFrame
    labels: np.ndarray
    condensed_edges: pd.DataFrame
    timing_info: dict


def chunked_hks_pipeline(
    mesh,
    query_indices=None,
    simplify_agg=7,
    simplify_target_reduction=0.7,
    overlap_distance=20_000,
    max_vertex_threshold=20_000,
    min_vertex_threshold=200,
    max_overlap_neighbors=40_000,
    n_components=32,
    t_min=5e4,
    t_max=2e7,
    max_eigenvalue=5e-6,
    robust=True,
    mollify_factor=1e-5,
    truncate_extra=True,
    drop_first=True,
    decomposition_dtype=np.float64,
    compute_hks_kwargs: dict = {},
    nuc_point=None,
    distance_threshold=3.0,
    auxiliary_features=True,
    n_jobs=-1,
    verbose=False,
) -> Result:
    """
    Compute the Heat Kernel Signature (HKS) on a mesh in a chunked fashion. Also
    aggregates HKS features in local regions of the mesh.

    Parameters
    ----------
    mesh :
        The input mesh. Should be a tuple of (vertices, faces), or an object with
        `vertices` and `faces` attributes.
    query_indices :
        The indices of the mesh of interest. If provided, will only compute the HKS for
        chunks which contain these indices. Otherwise, will compute the HKS for all
        chunks.
    simplify_agg :
        Controls how aggressively to decimate the mesh. A value of 10 will result in a
        fast decimation at the expense of mesh quality and shape. A value of 0 will
        attempt to preserve the original mesh geometry at the expense of time. Setting
        a low value may result in being unable to reach the `target_reduction`.
    simplify_target_reduction :
        The target reduction for the mesh simplification. Fraction of the original mesh
        to remove. If set to 0.9, this function will try to reduce the data set to 10%
        of its original size and will remove 90% of the input triangles.
    overlap_distance :
        The geodesic distance to overlap mesh chunks. Larger values will result in less
        distortion (especially for long timescales) but more computation time.
    max_vertex_threshold :
        The maximum number of vertices for a mesh chunk, before overlapping.
    min_vertex_threshold :
        The minimum number of vertices for a mesh chunk to be included in subsequent
        computations. This can be used to filter out small disconnected pieces of the
        mesh.
    max_overlap_neighbors :
        The maximum number of neighbors to consider when overlapping mesh chunks. This
        overrules `overlap_distance`.
    n_components :
        The number of timescales for the HKS computation. This determines the
        number of HKS features. Timescales will be logarithmically spaced between
        `t_min` and `t_max`.
    t_min :
        The minimum timescale for the HKS computation.
    t_max :
        The maximum timescale for the HKS computation.
    max_eval :
        The maximum eigenvalue for the HKS computation. Larger eigenvalues give a more
        detailed HKS result at the expense of computation time.
    robust :
        Whether to use the robust laplacian for the HKS computation. This is generally
        recommended as it does a better job of handling degenerate meshes.
    mollify_factor :
        The mollification factor for the robust laplacian computation.
    truncate_extra :
        Whether to truncate extra eigenpairs that may be computed past `max_eigenvalue`.
    drop_first :
        Whether to drop the first eigenpair from the computation, which will always be
        proportional to the areas of each vertex.
    nuc_point:
        The coordinates of the nucleus point. If provided, will compute the distance
        from each vertex to this point and include it as a feature.
    distance_threshold:
        The distance threshold for agglomerating the mesh. This is used to
        determine which chunks to merge together based on their HKS features.
    n_jobs :
        The number of jobs to use for the computation. See `joblib.Parallel`.
    verbose :
        Whether to print verbose output.

    Returns
    -------
    :
        The mesh after simplification.
    :
        The mapping from the original mesh to the simplified mesh. Has length of the
        number of vertices in the original mesh; each element contains the index that
        the vertex maps to in the simplified mesh.
    :
        The mesh stitcher object. Contains information about the mesh chunks and their
        overlaps.
    :
        The HKS features for each vertex in the simplified mesh.
    :
        The aggregated HKS features for each domain in the mesh.
    :
        The labels for each vertex in the simplified mesh, indicating which domain it
        belongs to.
    :
        Timing information for each step of the pipeline.

    Notes
    -----
    This pipeline consists of the following steps:

        1. Mesh simplification, using https://github.com/pyvista/fast-simplification.
        2. Mesh splitting, using a routine which iteratively does spectral bisection of
           the mesh until all chunks are below the `max_vertex_threshold`. These chunks
           are then grown to overlap using the `overlap_distance` parameter.
        3. Computation of the heat kernel signature of Sun et al (2008). This routine
           uses the robust laplacian of Crane et al. (2020) for more stable results. It
           also leverages the band-by-band eigensolver method of Vallet and Levy (2008).
        4. Agglomeration of the mesh into local domains which are bounded in the
           variance of the HKS features. This uses the implementation of Ward's method
           in scikit-learn which allows for a connectivity constraint.
        5. Aggregation of the computed features to the local domains. This takes the
           area-weighted mean of the features for each domain.
    """
    timing_info = {}

    # input mesh
    original_mesh = interpret_mesh(mesh)

    mesh, indices_from_original = threshold_mesh_by_component_size(
        original_mesh, size_threshold=min_vertex_threshold
    )
    # TODO need to somehow handle the case where the mesh is empty after thresholding

    # mesh simplification
    # NOTE: for some reason the order here differs from that in replay_simplification,
    # we want the latter so as to preserve the indices for `mapping`
    if simplify_target_reduction is not None:
        _, _, collapses = simplify(
            mesh[0],
            mesh[1],
            agg=simplify_agg,
            target_reduction=simplify_target_reduction,
            return_collapses=True,
        )

        vertices, faces, thresh_to_simple_mapping = replay_simplification(
            points=mesh[0],
            triangles=mesh[1],
            collapses=collapses,
        )
        mesh = (vertices, faces)
    else:
        thresh_to_simple_mapping = np.arange(len(mesh[0]))

    # mesh splitting
    currtime = time.time()
    stitcher = MeshStitcher(mesh, n_jobs=n_jobs, verbose=verbose)
    stitcher.split_mesh(
        overlap_distance=overlap_distance,
        max_vertex_threshold=max_vertex_threshold,
        min_vertex_threshold=min_vertex_threshold,
        max_overlap_neighbors=max_overlap_neighbors,
        verify_connected=False,
    )
    timing_info["split_time"] = time.time() - currtime

    # compute HKS
    currtime = time.time()
    if verbose:
        print("Computing HKS across submeshes...")
    if query_indices is None:
        X_hks = stitcher.apply(
            compute_hks,
            n_components=n_components,
            t_min=t_min,
            t_max=t_max,
            max_eigenvalue=max_eigenvalue,
            robust=robust,
            mollify_factor=mollify_factor,
            truncate_extra=truncate_extra,
            drop_first=drop_first,
            decomposition_dtype=decomposition_dtype,
            **compute_hks_kwargs,
        )
    else:
        X_hks = stitcher.subset_apply(
            compute_hks,
            query_indices,
            reindex=False,
            n_components=n_components,
            t_min=t_min,
            t_max=t_max,
            max_eigenvalue=max_eigenvalue,
            robust=robust,
            mollify_factor=mollify_factor,
            truncate_extra=truncate_extra,
            drop_first=drop_first,
            decomposition_dtype=decomposition_dtype,
            **compute_hks_kwargs,
        )
    with np.errstate(divide="ignore"):
        log_X_hks = np.log(X_hks)
    X_hks_df = pd.DataFrame(X_hks, columns=[f"hks_{i}" for i in range(X_hks.shape[1])])
    timing_info["hks_time"] = time.time() - currtime

    # Non-HKS features
    # currtime = time.time()
    # aux_X = []
    # aux_X_features = []

    # component_sizes = component_size_transform(mesh, mesh_indices)
    # aux_X.append(component_sizes)
    # aux_X_features.append("component_size")

    # if nuc_point is not None:
    #     distances_to_nuc = compute_distances_to_point(mesh[0], nuc_point)
    # else:
    #     distances_to_nuc = np.full(mesh[0].shape[0], np.nan)
    # aux_X.append(distances_to_nuc)
    # aux_X_features.append("distance_to_nucleus")

    # aux_X = np.column_stack(aux_X)
    # aux_X_df = pd.DataFrame(aux_X, columns=aux_X_features)
    # timing_info["aux_time"] = time.time() - currtime

    # joined_X_df = pd.concat([X_hks_df, aux_X_df], axis=1)

    joined_X_df = X_hks_df

    # agglomeration of mesh to domains
    currtime = time.time()

    simple_agg_labels = agglomerate_split_mesh(
        stitcher, log_X_hks, distance_thresholds=distance_threshold
    )
    timing_info["agglomeration_time"] = time.time() - currtime

    # aggregate features
    currtime = time.time()
    areas = compute_vertex_areas(mesh)
    agg_features_df = aggregate_features(
        joined_X_df, simple_agg_labels, func="mean", weights=areas
    )
    agg_features_df = np.log(agg_features_df)
    timing_info["aggregation_time"] = time.time() - currtime

    # reconstruct mapping to original mesh
    mapping = np.full(len(original_mesh[0]), -1, dtype=np.int32)
    mapping[indices_from_original] = thresh_to_simple_mapping

    agg_labels = expand_labels(simple_agg_labels, mapping)

    # condense mesh to graph, compute some additional auxiliary features
    condensed_node_table, condensed_edge_table = condense_mesh_to_graph(
        original_mesh, agg_labels, add_component_features=True
    )

    if auxiliary_features:
        if nuc_point is not None:
            condensed_node_table["distance_to_nucleus"] = compute_distances_to_point(
                condensed_node_table[["x", "y", "z"]].values, nuc_point
            )
        else:
            condensed_node_table["distance_to_nucleus"] = np.nan

        agg_features_df = pd.concat(
            [agg_features_df, condensed_node_table],
            axis=1,
        )

    out = Result(
        mesh,
        mapping,
        stitcher,
        joined_X_df,
        simple_agg_labels,
        agg_features_df,
        agg_labels,
        condensed_edge_table,
        timing_info,
    )

    return out


class CondensedHKSResult(NamedTuple):
    simple_mesh: tuple
    mapping: np.ndarray
    stitcher: MeshStitcher
    simple_labels: np.ndarray
    labels: np.ndarray
    condensed_features: pd.DataFrame
    condensed_nodes: pd.DataFrame
    condensed_edges: pd.DataFrame
    timing_info: dict


def compute_condensed_hks(
    mesh,
    n_components=32,
    t_min=5e4,
    t_max=2e7,
    max_eigenvalue=5e-6,
    robust=True,
    mollify_factor=1e-5,
    truncate_extra=True,
    drop_first=True,
    decomposition_dtype=np.float32,
    compute_hks_kwargs: dict = {},
    distance_threshold=3.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    X_hks = compute_hks(
        mesh,
        n_components=n_components,
        t_min=t_min,
        t_max=t_max,
        max_eigenvalue=max_eigenvalue,
        robust=robust,
        mollify_factor=mollify_factor,
        truncate_extra=truncate_extra,
        drop_first=drop_first,
        decomposition_dtype=decomposition_dtype,
        **compute_hks_kwargs,
    )

    with np.errstate(divide="ignore"):
        log_X_hks = np.log(X_hks)

    agg_labels = agglomerate_mesh(
        mesh,
        log_X_hks,
        distance_thresholds=distance_threshold,
    )

    weights = compute_vertex_areas(mesh)
    X_hks_condensed = aggregate_features(
        pd.DataFrame(X_hks, columns=[f"hks_{i}" for i in range(X_hks.shape[1])]),
        agg_labels,
        func="mean",
        weights=weights,
    )

    return X_hks_condensed, agg_labels


def condensed_hks_pipeline(
    mesh,
    simplify_agg=7,
    simplify_target_reduction=0.7,
    overlap_distance=20_000,
    max_vertex_threshold=20_000,
    min_vertex_threshold=200,
    max_overlap_neighbors=40_000,
    n_components=32,
    t_min=5e4,
    t_max=2e7,
    max_eigenvalue=5e-6,
    robust=True,
    mollify_factor=1e-5,
    truncate_extra=True,
    drop_first=True,
    decomposition_dtype=np.float32,
    compute_hks_kwargs: dict = {},
    nuc_point=None,
    distance_threshold=3.0,
    auxiliary_features=True,
    n_jobs=-1,
    verbose=False,
) -> CondensedHKSResult:
    """
    Compute the Heat Kernel Signature (HKS) on a mesh in a chunked fashion. Also
    aggregates HKS features in local regions of the mesh.

    Parameters
    ----------
    mesh :
        The input mesh. Should be a tuple of (vertices, faces), or an object with
        `vertices` and `faces` attributes.
    query_indices :
        The indices of the mesh of interest. If provided, will only compute the HKS for
        chunks which contain these indices. Otherwise, will compute the HKS for all
        chunks.
    simplify_agg :
        Controls how aggressively to decimate the mesh. A value of 10 will result in a
        fast decimation at the expense of mesh quality and shape. A value of 0 will
        attempt to preserve the original mesh geometry at the expense of time. Setting
        a low value may result in being unable to reach the `target_reduction`.
    simplify_target_reduction :
        The target reduction for the mesh simplification. Fraction of the original mesh
        to remove. If set to 0.9, this function will try to reduce the data set to 10%
        of its original size and will remove 90% of the input triangles.
    overlap_distance :
        The geodesic distance to overlap mesh chunks. Larger values will result in less
        distortion (especially for long timescales) but more computation time.
    max_vertex_threshold :
        The maximum number of vertices for a mesh chunk, before overlapping.
    min_vertex_threshold :
        The minimum number of vertices for a mesh chunk to be included in subsequent
        computations. This can be used to filter out small disconnected pieces of the
        mesh.
    max_overlap_neighbors :
        The maximum number of neighbors to consider when overlapping mesh chunks. This
        overrules `overlap_distance`.
    n_components :
        The number of timescales for the HKS computation. This determines the
        number of HKS features. Timescales will be logarithmically spaced between
        `t_min` and `t_max`.
    t_min :
        The minimum timescale for the HKS computation.
    t_max :
        The maximum timescale for the HKS computation.
    max_eval :
        The maximum eigenvalue for the HKS computation. Larger eigenvalues give a more
        detailed HKS result at the expense of computation time.
    robust :
        Whether to use the robust laplacian for the HKS computation. This is generally
        recommended as it does a better job of handling degenerate meshes.
    mollify_factor :
        The mollification factor for the robust laplacian computation.
    truncate_extra :
        Whether to truncate extra eigenpairs that may be computed past `max_eigenvalue`.
    drop_first :
        Whether to drop the first eigenpair from the computation, which will always be
        proportional to the areas of each vertex.
    nuc_point:
        The coordinates of the nucleus point. If provided, will compute the distance
        from each vertex to this point and include it as a feature.
    distance_threshold:
        The distance threshold for agglomerating the mesh. This is used to
        determine which chunks to merge together based on their HKS features.
    n_jobs :
        The number of jobs to use for the computation. See `joblib.Parallel`.
    verbose :
        Whether to print verbose output.

    Returns
    -------
    :
        The mesh after simplification.
    :
        The mapping from the original mesh to the simplified mesh. Has length of the
        number of vertices in the original mesh; each element contains the index that
        the vertex maps to in the simplified mesh.
    :
        The mesh stitcher object. Contains information about the mesh chunks and their
        overlaps.
    :
        The HKS features for each vertex in the simplified mesh.
    :
        The aggregated HKS features for each domain in the mesh.
    :
        The labels for each vertex in the simplified mesh, indicating which domain it
        belongs to.
    :
        Timing information for each step of the pipeline.

    Notes
    -----
    This pipeline consists of the following steps:

        1. Mesh simplification, using https://github.com/pyvista/fast-simplification.
        2. Mesh splitting, using a routine which iteratively does spectral bisection of
           the mesh until all chunks are below the `max_vertex_threshold`. These chunks
           are then grown to overlap using the `overlap_distance` parameter.
        3. Computation of the heat kernel signature of Sun et al (2008). This routine
           uses the robust laplacian of Crane et al. (2020) for more stable results. It
           also leverages the band-by-band eigensolver method of Vallet and Levy (2008).
        4. Agglomeration of the mesh into local domains which are bounded in the
           variance of the HKS features. This uses the implementation of Ward's method
           in scikit-learn which allows for a connectivity constraint.
        5. Aggregation of the computed features to the local domains. This takes the
           area-weighted mean of the features for each domain.
    """
    timing_info = {}
    starttime = time.time()

    # input mesh
    original_mesh = interpret_mesh(mesh)

    mesh, indices_from_original = threshold_mesh_by_component_size(
        original_mesh, size_threshold=min_vertex_threshold
    )

    # mesh simplification
    # NOTE: for some reason the order here differs from that in replay_simplification,
    # we want the latter so as to preserve the indices for `mapping`
    if simplify_target_reduction is not None:
        _, _, collapses = simplify(
            mesh[0],
            mesh[1],
            agg=simplify_agg,
            target_reduction=simplify_target_reduction,
            return_collapses=True,
        )

        vertices, faces, thresh_to_simple_mapping = replay_simplification(
            points=mesh[0],
            triangles=mesh[1],
            collapses=collapses,
        )
        mesh = (vertices, faces)
    else:
        thresh_to_simple_mapping = np.arange(len(mesh[0]))

    # mesh splitting
    currtime = time.time()
    stitcher = MeshStitcher(mesh, n_jobs=n_jobs, verbose=verbose)
    stitcher.split_mesh(
        overlap_distance=overlap_distance,
        max_vertex_threshold=max_vertex_threshold,
        min_vertex_threshold=min_vertex_threshold,
        max_overlap_neighbors=max_overlap_neighbors,
        verify_connected=False,
    )
    timing_info["split_time"] = time.time() - currtime

    # compute HKS
    currtime = time.time()
    if verbose:
        print("Computing HKS across submeshes...")

    results_by_submesh = stitcher.apply(
        compute_condensed_hks,
        n_components=n_components,
        t_min=t_min,
        t_max=t_max,
        max_eigenvalue=max_eigenvalue,
        robust=robust,
        mollify_factor=mollify_factor,
        truncate_extra=truncate_extra,
        drop_first=drop_first,
        decomposition_dtype=decomposition_dtype,
        compute_hks_kwargs=compute_hks_kwargs,
        distance_threshold=distance_threshold,
        stitch=False,
    )
    sub_agg_labels = stitcher.stitch_features(
        [result[1] for result in results_by_submesh],
        fill_value=-1,
    ).reshape(-1)
    data_by_submesh = [res[0] for res in results_by_submesh]

    simple_agg_labels, condensed_hks_df = fix_split_labels_and_features(
        sub_agg_labels,
        stitcher.submesh_mapping,
        data_by_submesh,
    )

    timing_info["hks_time"] = time.time() - currtime

    condensed_hks_df = np.log(condensed_hks_df)

    # reconstruct mapping to original mesh
    mapping = np.full(len(original_mesh[0]), -1, dtype=np.int32)
    mapping[indices_from_original] = thresh_to_simple_mapping

    agg_labels = expand_labels(simple_agg_labels, mapping)

    # condense mesh to graph, compute some additional auxiliary features
    condensed_node_table, condensed_edge_table = condense_mesh_to_graph(
        original_mesh, agg_labels, add_component_features=True
    )

    if auxiliary_features:
        if nuc_point is not None:
            condensed_node_table["distance_to_nucleus"] = compute_distances_to_point(
                condensed_node_table[["x", "y", "z"]].values, nuc_point
            ).astype(np.float32)
        else:
            condensed_node_table["distance_to_nucleus"] = np.empty(
                condensed_node_table.shape[0], dtype=np.float32
            )

    # for consistency with the rest of the pipeline, make sure null label is present
    assert -1 in condensed_hks_df.index

    timing_info["pipeline_time"] = time.time() - starttime
    out = CondensedHKSResult(
        mesh,
        mapping,
        stitcher,
        simple_agg_labels,
        agg_labels,
        condensed_hks_df,
        condensed_node_table,
        condensed_edge_table,
        timing_info,
    )

    return out
