import time

from .decompose import compute_hks
from .split import MeshStitcher
from .types import interpret_mesh


def chunked_hks_pipeline(
    mesh,
    mesh_indices=None,
    n_scales=64,
    t_min=5e4,
    t_max=2e7,
    max_eval=5e-6,
    overlap_distance=20_000,
    max_vertex_threshold=20_000,
    min_vertex_threshold=50,
    robust=True,
    mollify_factor=1e-5,
    truncate_extra=True,
    reindex=True,
    n_jobs=-1,
    verbose=False,
    return_timing=False,
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
    mesh = interpret_mesh(mesh)
    currtime = time.time()
    stitcher = MeshStitcher(mesh, n_jobs=n_jobs, verbose=verbose)
    stitcher.split_mesh(
        overlap_distance=overlap_distance,
        max_vertex_threshold=max_vertex_threshold,
        min_vertex_threshold=min_vertex_threshold,
    )
    split_time = time.time() - currtime
    currtime = time.time()
    if verbose:
        print("Computing HKS across submeshes...")
    if mesh_indices is None:
        X = stitcher.apply(
            compute_hks,
            n_scales=n_scales,
            t_min=t_min,
            t_max=t_max,
            max_eigenvalue=max_eval,
            robust=robust,
            mollify_factor=mollify_factor,
            truncate_extra=truncate_extra,
        )
    else:
        X = stitcher.subset_apply(
            compute_hks,
            mesh_indices,
            reindex=reindex,
            n_scales=n_scales,
            t_min=t_min,
            t_max=t_max,
            max_eigenvalue=max_eval,
            robust=robust,
            mollify_factor=mollify_factor,
            truncate_extra=truncate_extra,
        )
    hks_time = time.time() - currtime

    timing_info = {
        "split_time": split_time,
        "hks_time": hks_time,
    }

    if return_timing:
        return X, timing_info
    else:
        return X
