import time
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sparse
from scipy.linalg import eigh
from tqdm.auto import tqdm

from .laplacian import area_matrix, cotangent_laplacian
from .types import ArrayLike, Mesh


def decompose_laplacian(
    L,
    M,
    n_components=100,
    op_inv=None,
    sigma=-1e-10,
    tol=1e-10,
    ncv=None,
    prefactor=None,
):
    """
    Solves the generalized eigenvalue problem.
    Change solver if necessary

    Parameters
    -----------------------------
    L:
        (n,n) - sparse matrix of cotangent weights
    M:
        (n,n) - sparse matrix of area weights
    n_components :
        int - number of eigenvalues to compute

    Returns
    -----------------------------
    eigenvalues   : np.ndarray
        (n_components,) - array of eigenvalues
    eigenvectors  : np.ndarray
        (n, n_components) - array of eigenvectors
    """
    if prefactor is not None:
        if prefactor == "lu":
            if not sparse.isspmatrix_csc(L):
                L = L.tocsc()
            lu = sparse.linalg.splu(L - sigma * M)
            op_inv = sparse.linalg.LinearOperator(
                matvec=lu.solve, shape=L.shape, dtype=L.dtype
            )
        # TODO add cholesky prefactor? tempted, but it adds a dependency and didn't seem
        # to change things much in terms of timing
    else:
        op_inv = None
    # k = n_components
    # n = L.shape[0]
    # ncv_factor = 1.5
    # ncv = min(n, max(ncv_factor * k + 1, 20))
    if n_components >= L.shape[0]:
        eigenvalues, eigenvectors = eigh(L.toarray(), M.toarray())
    else:
        eigenvalues, eigenvectors = sparse.linalg.eigsh(
            L, k=n_components, M=M, sigma=sigma, OPinv=op_inv, tol=tol, ncv=ncv
        )
    return eigenvalues, eigenvectors


def decompose_mesh(
    mesh: Mesh,
    n_components=100,
    op_inv=None,
    sigma=-1e-10,
    tol=1e-10,
    prefactor=None,
):
    L = cotangent_laplacian(mesh)
    M = area_matrix(mesh)
    return decompose_laplacian(
        L,
        M,
        n_components=n_components,
        op_inv=op_inv,
        sigma=sigma,
        tol=tol,
        prefactor=prefactor,
    )


def decompose_laplacian_by_bands(
    L,
    M,
    max_eigenvalue=1e-9,
    band_size=50,
    truncate_extra=True,
    verbose=False,
):
    # REF: section 4.1 of Spectral Mesh Processing, Levy & Zhang 2009
    # The idea is that because ARAPACK is good at solving for large eigenvalues, or,
    # eigenvalues near sigma for the shift-invert mode, we get a speedup from solving
    # for bands of eigenvalues at a time, where in each band we are close to some sigma.
    # Also has the advantage of being able to (roughly) specify a max eigenvalue to
    # compute up to, since we'll only overshoot by at most 1 band.
    eigenvalues = []
    eigenvectors = []
    band_max_eigenvalue = 0
    sigma = 0
    # n_steps = 1000
    # approx_range = np.linspace(0, max_eigenvalue, n_steps)
    pbar = tqdm(total=max_eigenvalue, disable=not verbose)
    while band_max_eigenvalue < max_eigenvalue:
        if verbose >= 2:
            print(f"Computing band with sigma={sigma:.3g}")
        band_eigenvalues, band_eigenvectors = decompose_laplacian(
            L, M, n_components=band_size, sigma=sigma
        )
        band_max_eigenvalue = np.max(band_eigenvalues)
        band_min_eigenvalue = np.min(band_eigenvalues)

        if len(eigenvalues) == 0:
            eigenvalues.extend(band_eigenvalues)
            eigenvectors.extend(band_eigenvectors.T)

            eigenvalue_bandwidth = band_max_eigenvalue - band_min_eigenvalue
            sigma = band_max_eigenvalue + 0.4 * eigenvalue_bandwidth
            pbar.update(band_max_eigenvalue)
        else:
            # find the index where the new eigenvalues are within the tolerance
            # of the last eigenvalue
            last_eigenvalue = eigenvalues[-1]
            diffs = np.abs(band_eigenvalues - last_eigenvalue)
            tol = 1e-16
            if np.min(diffs) > tol:
                # retry with a smaller sigma
                sigma = sigma - 0.2 * eigenvalue_bandwidth
                if verbose >= 2:
                    print(f"Will retry band with sigma={sigma:.3g}")
            else:
                # save the results of this band
                closest_idx = np.argmin(diffs)
                eigenvalues.extend(band_eigenvalues[closest_idx + 1 :])
                eigenvectors.extend(band_eigenvectors[:, closest_idx + 1 :].T)

                # Continue on to the next band

                # This is the heuristic suggested in Levy & Zhang 2009 for choosing sigma to be
                # roughly in the middle of a band that still overlaps what we've already seen.
                # It looked to work quite well in practice.

                eigenvalue_bandwidth = band_max_eigenvalue - band_min_eigenvalue
                sigma = band_max_eigenvalue + 0.4 * eigenvalue_bandwidth
                pbar.update(band_max_eigenvalue - last_eigenvalue)

    pbar.close()

    eigenvalues = np.array(eigenvalues)
    eigenvectors = np.stack(eigenvectors, axis=1)

    if truncate_extra:
        # Truncate to the max_eigenvalue
        truncation_idx = np.searchsorted(eigenvalues, max_eigenvalue)
        eigenvalues = eigenvalues[: truncation_idx + 1]
        eigenvectors = eigenvectors[:, : truncation_idx + 1]

    return eigenvalues, eigenvectors


def compute_hks_old(
    mesh: Mesh,
    max_eigenvalue: float = 1e-8,
    t_max: Optional[float] = None,
    t_min: Optional[float] = None,
    n_scales: int = 32,
    scales: Optional[ArrayLike] = None,
    band_size=50,
    truncate_extra=False,
    # robust_eps=None,
    # fix=False,
    robust=False,
    mollify_factor=1e-5,
    verbose=False,
):
    # TODO unify this with the code above, mostly redundant!

    L, M = cotangent_laplacian(mesh, robust=robust, mollify_factor=mollify_factor)

    eigenvalues = []
    band_max_eigenvalue = 0
    sigma = -0.000001
    last_eigenvalue = 0
    eigenvalue_bandwidth = 0
    if scales is not None:
        n_scales = len(scales)
    if t_max is not None and t_min is not None:
        scales = np.geomspace(t_min, t_max, n_scales)
    hks = np.zeros((L.shape[0], n_scales))
    timing = {}
    timing["decompose"] = 0
    timing["band_hks"] = 0
    timing["sum"] = 0
    pbar = tqdm(total=max_eigenvalue, disable=not verbose)
    while band_max_eigenvalue < max_eigenvalue:
        if verbose >= 2:
            print(f"Computing band with sigma={sigma:.3g}")

        currtime = time.time()
        band_eigenvalues, band_eigenvectors = decompose_laplacian(
            L, M, n_components=band_size, sigma=sigma
        )
        timing["decompose"] += time.time() - currtime

        if band_max_eigenvalue == 0 and scales is None:
            min_eigenvalue = band_eigenvalues[1]  # skip the first eigenvalue, is 0
            t_min = 4 * np.log(10) / max_eigenvalue
            t_max = 4 * np.log(10) / min_eigenvalue
            scales = np.geomspace(t_min, t_max, n_scales)

        # find the index where the new eigenvalues are within the tolerance
        # of the last eigenvalue
        diffs = np.abs(band_eigenvalues - last_eigenvalue)
        tol = 1e-16
        if np.min(diffs) > tol:
            # retry with a smaller sigma
            sigma = sigma - 0.2 * eigenvalue_bandwidth
            if verbose >= 2:
                print(f"Will retry band with sigma={sigma:.3g}")
            band_eigenvalues = None
            band_eigenvectors = None
            continue
        else:
            # get the non-overlapping part of this band
            closest_idx = np.argmin(diffs)
            band_eigenvalues = band_eigenvalues[closest_idx + 1 :]
            band_eigenvectors = band_eigenvectors[:, closest_idx + 1 :]

        # TODO could include other signatures/functions here
        # if method == "heat":
        # elif method == "wave":
        #     sigma = 7 * (np.max(scales) - np.min(scales)) / len(scales)
        #     band_coefs = np.exp(
        #         -np.square(scales[:, None] - np.log(np.abs(band_eigenvalues))[None, :])
        #         / (2 * sigma**2)
        #     )

        if truncate_extra and (band_eigenvalues[-1] > max_eigenvalue):
            # Truncate to the max_eigenvalue
            truncation_idx = np.searchsorted(band_eigenvalues, max_eigenvalue)
            band_eigenvalues = band_eigenvalues[: truncation_idx + 1]
            band_eigenvectors = band_eigenvectors[:, : truncation_idx + 1]

        currtime = time.time()
        # compute HKS
        band_coefs = np.exp(-np.outer(scales, band_eigenvalues))

        band_hks = np.einsum("tk,nk->nt", band_coefs, np.square(band_eigenvectors))
        timing["band_hks"] += time.time() - currtime

        currtime = time.time()
        hks += band_hks
        timing["sum"] += time.time() - currtime

        # update values for next iteration
        eigenvalues.extend(band_eigenvalues)
        band_max_eigenvalue = np.max(band_eigenvalues)
        band_min_eigenvalue = np.min(band_eigenvalues)
        eigenvalue_bandwidth = band_max_eigenvalue - band_min_eigenvalue
        sigma = band_max_eigenvalue + 0.4 * eigenvalue_bandwidth

        # update by the amount the max eigenvalue increased
        pbar.update(band_max_eigenvalue - last_eigenvalue)
        last_eigenvalue = band_eigenvalues[-1]

    pbar.close()

    # if fix:
    #     hks = hks[fixed_to_original_indices]
    #     hks[fixed_to_original_indices == -1] = np.nan

    if verbose >= 2:
        print("Timing:")
        total_time = sum(timing.values())
        for key, value in timing.items():
            print(f"{key}: {value:.3f} ({value / total_time:.2%})")

    return hks


def get_hks_filter(
    t_max: Optional[float] = None,
    t_min: Optional[float] = None,
    n_scales: int = 32,
) -> Callable:
    scales = np.geomspace(t_min, t_max, n_scales)

    def hks_filter(eigenvalues):
        coefs = np.exp(-np.outer(scales, eigenvalues))
        return coefs

    return hks_filter


def spectral_geometry_filter(
    mesh: Mesh,
    filter: Callable,
    max_eigenvalue: float = 1e-8,
    band_size: int = 50,
    truncate_extra: bool = True,
    drop_first: bool = True,
    robust: bool = True,
    mollify_factor: bool = 1e-5,
    verbose: int = False,
):
    """Apply a spectral filter to the geometry of a mesh.

    Parameters
    ----------
    mesh :
        The input mesh. Must be a tuple of vertices and faces as arrays, or be an object
        with a `vertices` and `faces` attribute.
    filter :
        A function that takes 1D array of eigenvalues, and returns a 2D array of filter
        coefficients, where the first dimension is the number of filters, and the second
        is the number of eigenvalues.
    max_eigenvalue :
        The maximum eigenvalue to compute the eigendecomposition up to.
    band_size :
        The number of eigenvalues to compute at a time using the band-by-band algorithm
        from [1]. This number should not affect the results, but may affect the speed.
    truncate_extra :
        If True, truncate the filter to the max_eigenvalue exactly. Due the the
        band-by-band algorithm, the filter may overshoot the max_eigenvalue by at most
        one band.
    drop_first :
        If True, drop the first eigenvalue and eigenvector. This should be 0 and the
        constant eigenvector scaled by vertex areas, so it is often not useful.
    robust :
        If True, use the robust laplacian computation described in [2].
    mollify_factor :
        The factor to use for the mollification when computing the robust laplacian.
        If robust is False, this parameter is ignored.
    verbose :
        If >0, print out additional information about the computation. Higher values
        give more information.

    Returns
    -------
    :
        A 2D array of features, where the first dimension is the number of vertices, and
        the second is the number of features.

    Notes
    -----
    Numerical errors are often due to a malformed mesh and therefore a malformed
    Laplacian. For this reason, it is recommended to use the robust Laplacian
    computation is used. Alternatively, make sure you are inputting a manifold mesh.

    References
    ----------
    [1] Spectral Geometry Processing with Manifold Harmonics, Vallet and Levy, 2008
    [2] A Laplacian for Nonmanifold Triangle Meshes, Sharp and Crane, 2020

    """

    # TODO add something about whether to throw out the first eigenpair
    # TODO look up whether the eigenvector should be x or M^{-1}x from the generalized
    # eigenvalue problem. In other words, dividing by the area. I saw something about
    # this in a paper and highlighted it and now I can't find

    L, M = cotangent_laplacian(mesh, robust=robust, mollify_factor=mollify_factor)

    eigenvalues = []
    band_max_eigenvalue = 0
    sigma = -0.000001
    last_eigenvalue = 0
    eigenvalue_bandwidth = 0

    # HACK: get the number of features for the filter
    n_features = filter([1, 2, 3]).shape[0]

    features = np.zeros((L.shape[0], n_features))
    timing = {}
    timing["decompose"] = 0
    timing["filter"] = 0
    timing["sum"] = 0
    pbar = tqdm(total=max_eigenvalue, disable=not verbose)
    while band_max_eigenvalue < max_eigenvalue:
        if verbose >= 2:
            print(f"Computing band with sigma={sigma:.3g}")

        currtime = time.time()
        band_eigenvalues, band_eigenvectors = decompose_laplacian(
            L, M, n_components=band_size, sigma=sigma
        )
        timing["decompose"] += time.time() - currtime

        # find the index where the new eigenvalues are within the tolerance
        # of the last eigenvalue
        diffs = np.abs(band_eigenvalues - last_eigenvalue)
        tol = 1e-16
        if np.min(diffs) > tol:
            # retry with a smaller sigma
            sigma = sigma - 0.2 * eigenvalue_bandwidth
            if verbose >= 2:
                print(f"Will retry band with sigma={sigma:.3g}")
            band_eigenvalues = None
            band_eigenvectors = None
            continue
        else:
            # get the non-overlapping part of this band
            closest_idx = np.argmin(diffs)
            band_eigenvalues = band_eigenvalues[closest_idx + 1 :]
            band_eigenvectors = band_eigenvectors[:, closest_idx + 1 :]

        if truncate_extra and (band_eigenvalues[-1] > max_eigenvalue):
            # Truncate to the max_eigenvalue
            truncation_idx = np.searchsorted(band_eigenvalues, max_eigenvalue)
            band_eigenvalues = band_eigenvalues[: truncation_idx + 1]
            band_eigenvectors = band_eigenvectors[:, : truncation_idx + 1]

        currtime = time.time()

        if drop_first and len(eigenvalues) == 0:
            first_idx = 1
        else:
            first_idx = 0

        # compute filter based on eigenvalues
        band_coefs = filter(band_eigenvalues[first_idx:])

        band_features = np.einsum(
            "tk,nk->nt", band_coefs, np.square(band_eigenvectors[:, first_idx:])
        )
        timing["filter"] += time.time() - currtime

        currtime = time.time()
        features += band_features
        timing["sum"] += time.time() - currtime

        # update values for next iteration
        eigenvalues.extend(band_eigenvalues)
        band_max_eigenvalue = np.max(band_eigenvalues)
        band_min_eigenvalue = np.min(band_eigenvalues)
        eigenvalue_bandwidth = band_max_eigenvalue - band_min_eigenvalue
        sigma = band_max_eigenvalue + 0.4 * eigenvalue_bandwidth

        # update by the amount the max eigenvalue increased
        pbar.update(band_max_eigenvalue - last_eigenvalue)
        last_eigenvalue = band_eigenvalues[-1]

    pbar.close()

    if verbose >= 2:
        print("Timing:")
        total_time = sum(timing.values())
        for key, value in timing.items():
            print(f"{key}: {value:.3f} ({value / total_time:.2%})")

    return features


def compute_hks(
    mesh: Mesh,
    max_eigenvalue: float = 1e-8,
    t_max: Optional[float] = None,
    t_min: Optional[float] = None,
    n_scales: int = 32,
    band_size: int = 50,
    truncate_extra: bool = False,
    drop_first: bool = False,
    robust: bool = True,
    mollify_factor: float = 1e-5,
    verbose: int = False,
):
    filter_func = get_hks_filter(t_max, t_min, n_scales)
    out = spectral_geometry_filter(
        mesh,
        filter_func,
        max_eigenvalue=max_eigenvalue,
        band_size=band_size,
        truncate_extra=truncate_extra,
        drop_first=drop_first,
        robust=robust,
        mollify_factor=mollify_factor,
        verbose=verbose,
    )
    return out
