from typing import Optional

import numpy as np
import scipy.sparse as sparse
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
    eigenvalues, eigenvectors = sparse.linalg.eigsh(
        L, k=n_components, M=M, sigma=sigma, OPinv=op_inv, tol=tol
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


def compute_hks(
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
    # if fix:
    #     if verbose:
    #         print("Fixing mesh")
    #     fixed_mesh = fix_mesh(mesh, remove_smallest_components=False, joincomp=True)
    #     vertex_indices = pd.MultiIndex.from_arrays(mesh[0].T)
    #     fixed_vertex_indices = pd.MultiIndex.from_arrays(fixed_mesh[0].T)
    #     fixed_to_original_indices = fixed_vertex_indices.get_indexer_for(vertex_indices)
    #     mesh = fixed_mesh

    # TODO unify this with the code above, mostly redundant!
    # L = cotangent_laplacian(mesh, robust_eps=robust_eps)
    # M = area_matrix(mesh)
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
    pbar = tqdm(total=max_eigenvalue, disable=not verbose)
    while band_max_eigenvalue < max_eigenvalue:
        if verbose >= 2:
            print(f"Computing band with sigma={sigma:.3g}")

        band_eigenvalues, band_eigenvectors = decompose_laplacian(
            L, M, n_components=band_size, sigma=sigma
        )

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

        # compute HKS
        band_coefs = np.exp(-np.outer(scales, band_eigenvalues))

        band_hks = np.einsum("tk,nk->nt", band_coefs, np.square(band_eigenvectors))
        hks += band_hks

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

    return hks