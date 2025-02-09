import time

import numpy as np
import pandas as pd
from fast_simplification import simplify

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
    Compute the Heat Kernel Signature (HKS) on a mesh in a chunked fashion.

    Parameters
    ----------
    mesh :
        The input mesh.
    mesh_indices :
        The indices of the mesh to compute the HKS on.
    n_scales :
        The number of scales for the HKS computation.
    t_min :
        The minimum timescale for the HKS computation.
    t_max :
        The maximum timescale for the HKS computation.
    max_eval :
        The maximum eigenvalue for the HKS computation. Larger eigenvalues give a more
        detailed HKS result at the expense of computation time.
    overlap_distance :
        The distance to overlap the mesh chunks. Larger values will result in less
        distortion (especially for long timescales) but more computation time.
    max_vertex_threshold :
        The maximum number of vertices for a mesh chunk, before overlapping.
    min_vertex_threshold :
        The minimum number of vertices for a mesh chunk, before overlapping.
    robust :
        Whether to use the robust laplacian for the HKS computation. This is generally
        recommended as it does a better job of handling degenerate meshes.
    mollify_factor :
        The mollification factor for the robust laplacian computation.
    truncate_extra :
        Whether to truncate extra eigenpairs that may be computed past `max_eigenvalue`.
    n_jobs :
        The number of jobs to use for the computation. See `joblib.Parallel`.
    verbose :
        Whether to print verbose output.
    return_timing :
        Whether to return the timing information for each process.

    Returns
    -------
    np.ndarray
        The HKS features for the mesh, potentially subset to the specified
        `mesh_indices`.
    """
    timing_info = {}

    # input mesh
    mesh = interpret_mesh(mesh)

    # mesh simplification
    mesh = simplify(
        mesh[0], mesh[1], agg=simplify_agg, target_reduction=simplify_target_reduction
    )
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

    return mesh, stitcher, joined_X_df, agg_features_df, agg_labels, timing_info
