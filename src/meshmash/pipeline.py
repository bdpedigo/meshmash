import time

import numpy as np
import pandas as pd
from fast_simplification import replay_simplification, simplify

from .agglomerate import agglomerate_split_mesh, aggregate_features
from .decompose import compute_hks
from .laplacian import compute_vertex_areas
from .split import MeshStitcher
from .types import interpret_mesh
from .utils import (
    component_size_transform,
    compute_distances_to_point,
)


def chunked_hks_pipeline(
    mesh,
    query_indices=None,
    simplify_agg=7,
    simplify_target_reduction=0.7,
    overlap_distance=20_000,
    max_vertex_threshold=20_000,
    min_vertex_threshold=200,
    n_scales=32,
    t_min=5e4,
    t_max=2e7,
    max_eval=5e-6,
    robust=True,
    mollify_factor=1e-5,
    truncate_extra=True,
    drop_first=True,
    nuc_point=None,
    distance_threshold=5.0,
    n_jobs=-1,
    verbose=False,
):
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
    n_scales :
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
    This pipeline currently consists of the following steps: 

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
    mesh = interpret_mesh(mesh)

    # mesh simplification
    vertices, faces, collapses = simplify(
        mesh[0],
        mesh[1],
        agg=simplify_agg,
        target_reduction=simplify_target_reduction,
        return_collapses=True,
    )

    _, _, mapping = replay_simplification(
        points=mesh[0],
        triangles=mesh[1],
        collapses=collapses,
    )
    mesh = (vertices, faces)
    mesh_indices = np.arange(mesh[0].shape[0])

    # mesh splitting
    currtime = time.time()
    stitcher = MeshStitcher(mesh, n_jobs=n_jobs, verbose=verbose)
    stitcher.split_mesh(
        overlap_distance=overlap_distance,
        max_vertex_threshold=max_vertex_threshold,
        min_vertex_threshold=min_vertex_threshold,
    )
    timing_info["split_time"] = time.time() - currtime

    # compute HKS
    currtime = time.time()
    if verbose:
        print("Computing HKS across submeshes...")
    if query_indices is None:
        X_hks = stitcher.apply(
            compute_hks,
            n_scales=n_scales,
            t_min=t_min,
            t_max=t_max,
            max_eigenvalue=max_eval,
            robust=robust,
            mollify_factor=mollify_factor,
            truncate_extra=truncate_extra,
            drop_first=drop_first,
        )
    else:
        X_hks = stitcher.subset_apply(
            compute_hks,
            query_indices,
            reindex=False,
            n_scales=n_scales,
            t_min=t_min,
            t_max=t_max,
            max_eigenvalue=max_eval,
            robust=robust,
            mollify_factor=mollify_factor,
            truncate_extra=truncate_extra,
            drop_first=drop_first,
        )
    log_X_hks = np.log(X_hks)
    X_hks_df = pd.DataFrame(X_hks, columns=[f"hks_{i}" for i in range(X_hks.shape[1])])
    timing_info["hks_time"] = time.time() - currtime

    # Non-HKS features
    currtime = time.time()
    aux_X = []
    aux_X_features = []

    component_sizes = component_size_transform(mesh, mesh_indices)
    aux_X.append(component_sizes)
    aux_X_features.append("component_size")

    if nuc_point is not None:
        distances_to_nuc = compute_distances_to_point(mesh[0], nuc_point)
        aux_X.append(distances_to_nuc)
        aux_X_features.append("distance_to_nucleus")

    aux_X = np.column_stack(aux_X)
    aux_X_df = pd.DataFrame(aux_X, columns=aux_X_features)
    timing_info["aux_time"] = time.time() - currtime

    joined_X_df = pd.concat([X_hks_df, aux_X_df], axis=1)

    # agglomeration of mesh to domains
    currtime = time.time()
    areas = compute_vertex_areas(mesh)
    agg_labels = agglomerate_split_mesh(
        stitcher, log_X_hks, distance_thresholds=distance_threshold
    )
    timing_info["agglomeration_time"] = time.time() - currtime

    # aggregate features
    currtime = time.time()
    agg_features_df = aggregate_features(
        joined_X_df, agg_labels, func="mean", weights=areas
    )
    timing_info["aggregation_time"] = time.time() - currtime

    return (
        mesh,
        mapping,
        stitcher,
        joined_X_df,
        agg_features_df,
        agg_labels,
        timing_info,
    )
