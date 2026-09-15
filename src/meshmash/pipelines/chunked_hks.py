import time
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from ..agglomerate import (
    agglomerate_split_mesh,
    aggregate_features,
)
from ..decompose import compute_hks
from ..graph import condense_mesh_to_graph
from ..laplacian import compute_vertex_areas
from ..simplify import simplify_mesh
from ..split import MeshStitcher
from ..types import interpret_mesh
from ..utils import (
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
    n_jobs: Optional[int] = -1,
    verbose=False,
) -> Result:
    """Compute HKS features and aggregate them across overlapping mesh chunks.

    .. deprecated::
        Prefer [condensed_hks_pipeline][meshmash.pipelines.condensed_hks.condensed_hks_pipeline], which is more memory-efficient
        because it aggregates features within each chunk before stitching.
        This function performs aggregation *after* all chunk features are
        stitched together, which requires holding the full-mesh feature
        matrix in memory.

    Runs the following pipeline:

    1. Optional mesh simplification via ``fast-simplification``.
    2. Spectral bisection of the mesh into overlapping chunks
       ([MeshStitcher][meshmash.split.MeshStitcher]).
    3. Computation of [compute_hks][meshmash.decompose.compute_hks] per chunk.
    4. Connectivity-constrained Ward agglomeration of vertices into local
       domains ([agglomerate_split_mesh][meshmash.agglomerate.agglomerate_split_mesh]).
    5. Area-weighted aggregation of HKS features to each domain
       ([aggregate_features][meshmash.agglomerate.aggregate_features]).

    Parameters
    ----------
    mesh :
        Input mesh.  Either a ``(vertices, faces)`` tuple or an object
        with ``vertices`` and ``faces`` attributes.
    query_indices :
        Vertex indices of interest.  If provided, only chunks containing
        these vertices are processed.  ``None`` processes all chunks.
    simplify_agg :
        Decimation aggressiveness (0–10).  Higher values are faster but
        reduce mesh quality.  Low values may prevent reaching
        ``simplify_target_reduction``.
    simplify_target_reduction :
        Fraction of triangles to remove during simplification.  ``None``
        skips simplification.
    overlap_distance :
        Geodesic radius used to grow each chunk into its overlap region.
        Larger values reduce boundary artefacts for long timescales but
        increase compute time.
    max_vertex_threshold :
        Maximum vertices per core chunk before overlapping.
    min_vertex_threshold :
        Minimum component size; smaller components are discarded.
    max_overlap_neighbors :
        Cap on overlap region size (number of nearest neighbours);
        overrides ``overlap_distance`` when set.
    n_components :
        Number of HKS timescales.  Timescales are log-spaced between
        ``t_min`` and ``t_max`` and determine the number of HKS features
        per vertex.
    t_min :
        Minimum diffusion timescale.
    t_max :
        Maximum diffusion timescale.
    max_eigenvalue :
        Maximum Laplacian eigenvalue used in the HKS computation.
    robust :
        If ``True``, use the robust Laplacian for HKS (recommended).
    mollify_factor :
        Mollification factor for the robust Laplacian.
    truncate_extra :
        If ``True``, discard eigenpairs that overshoot ``max_eigenvalue``.
    drop_first :
        If ``True``, drop the first (area-proportional) eigenpair.
    decomposition_dtype :
        Floating-point dtype for the eigendecomposition.
    compute_hks_kwargs :
        Extra keyword arguments forwarded to
        [compute_hks][meshmash.decompose.compute_hks].
    nuc_point :
        Coordinates of the nucleus/reference point.  If provided, a
        ``distance_to_nucleus`` column is added to the condensed node table.
    distance_threshold :
        Ward linkage-distance threshold used to cut the agglomeration tree
        into local domains.
    auxiliary_features :
        If ``True``, append condensed node-table columns (centroid, area,
        etc.) to the aggregated feature DataFrame.
    n_jobs :
        Number of parallel workers for [Parallel][joblib.Parallel].  ``-1``
        uses all available cores.
    verbose :
        Verbosity level.

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
    mesh, thresh_to_simple_mapping = simplify_mesh(
        mesh, agg=simplify_agg, target_reduction=simplify_target_reduction
    )

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
