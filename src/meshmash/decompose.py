import time
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sparse
from scipy.interpolate import BSpline
from scipy.linalg import eigh
from scipy.sparse import coo_array, csc_array, csr_array
from tqdm.auto import tqdm

from .laplacian import cotangent_laplacian
from .types import Mesh


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
    indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    return eigenvalues, eigenvectors


def decompose_mesh(
    mesh: Mesh,
    n_components=100,
    op_inv=None,
    sigma=-1e-10,
    tol=1e-10,
    robust=True,
    mollify_factor=1e-5,
    prefactor=None,
):
    L, M = cotangent_laplacian(mesh, robust=robust, mollify_factor=mollify_factor)
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


def get_hks_filter(
    t_max: Optional[float] = None,
    t_min: Optional[float] = None,
    n_scales: int = 32,
    dtype: np.dtype = np.float64,
) -> Callable:
    scales = np.geomspace(t_min, t_max, n_scales, dtype=dtype)

    def hks_filter(eigenvalues):
        coefs = np.exp(-np.outer(scales, eigenvalues))
        return coefs

    return hks_filter


def construct_bspline_basis(e_min: float, e_max: float, n_components: int):
    extrapolate = False
    basis_degree = 3
    domain = np.array([e_min, e_max])

    width = (domain[1] - domain[0]) / (n_components + basis_degree - 1)

    t = np.linspace(
        domain[0] - width * (basis_degree - 1),
        domain[1] + width * (basis_degree - 1),
        n_components + basis_degree,
    )

    bases = []
    for shift in range(n_components):
        knots = t[shift : shift + basis_degree + 1]
        b = BSpline.basis_element(knots, extrapolate=extrapolate)
        bases.append(b)
    return bases


def construct_bspline_filter(e_min: float, e_max: float, n_components: int):
    bases = construct_bspline_basis(e_min, e_max, n_components)

    def bspline_filter(eigenvalues):
        coefs = np.stack([b(eigenvalues) for b in bases])
        coefs[~np.isfinite(coefs)] = 0
        return coefs

    return bspline_filter


def spectral_geometry_filter(
    mesh: Mesh,
    filter: Optional[Callable] = None,
    max_eigenvalue: float = 1e-8,
    band_size: int = 50,
    truncate_extra: bool = True,
    drop_first: bool = True,
    decomposition_dtype: Optional[np.dtype] = np.float64,
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
        is the number of eigenvalues. If None, the eigenvectors and eigenvalues
        themselves will be returned, and no filtering will be applied.
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
    if isinstance(mesh, tuple):
        if isinstance(mesh[0], (csr_array, csc_array, coo_array)):
            L, M = mesh
        else:
            raise ValueError(
                "If mesh is a tuple, it must be a tuple of (L, M) where L is the Laplacian "
                "and M is the mass matrix (or None)."
            )
    else:
        L, M = cotangent_laplacian(mesh, robust=robust, mollify_factor=mollify_factor)

    if decomposition_dtype is not None:
        L = L.astype(decomposition_dtype)
        if M is not None:
            M = M.astype(decomposition_dtype)

    if decomposition_dtype == np.float32 or decomposition_dtype == "float32":
        tol = 1e-8
        # this suprisingly didn't make much difference in time to go lower here
        eigen_tol = 1e-7
    elif decomposition_dtype == np.float64 or decomposition_dtype == "float64":
        # tol = 1e-16
        tol = 1e-12
        eigen_tol = 1e-12
    else:
        raise ValueError(f"Unknown decomposition_dtype: {decomposition_dtype}")

    eigenvalues = []
    band_max_eigenvalue = 0
    sigma = -1e-10
    last_eigenvalue = 0
    eigenvalue_bandwidth = 0

    if filter is not None:
        # HACK: get the number of features for the filter
        n_features = filter([1, 2, 3]).shape[0]
        features = np.zeros((L.shape[0], n_features), dtype=decomposition_dtype)
    else:
        # will just store the eigenvectors themselves
        features = []

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
            L, M, n_components=band_size, sigma=sigma, tol=eigen_tol
        )
        timing["decompose"] += time.time() - currtime

        # find the index where the new eigenvalues are within the tolerance
        # of the last eigenvalue
        diffs = np.abs(band_eigenvalues - last_eigenvalue)
        if (np.min(diffs)) > tol and (len(eigenvalues) > 0):  # ignore if 1st
            # retry with a smaller sigma
            sigma = sigma - 0.2 * eigenvalue_bandwidth
            if verbose >= 2:
                print(f"Will retry band with sigma={sigma:.3g}")
            band_eigenvalues = None
            band_eigenvectors = None
            continue
        elif len(eigenvalues) == 0:
            # this is the first band, so we can just use it as is
            pass
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

        # TODO: not sure if necessary, but for now, going to keep this part of the
        # algo in the original dtype
        # if band_eigenvalues.dtype != original_dtype:
        #     band_eigenvalues = band_eigenvalues.astype(original_dtype)
        #     band_eigenvectors = band_eigenvectors.astype(original_dtype)

        if filter is not None:
            # compute filter based on eigenvalues
            band_coefs = filter(band_eigenvalues[first_idx:])

            band_features = np.einsum(
                "tk,nk->nt",
                band_coefs,
                np.square(band_eigenvectors[:, first_idx:]),
                dtype=decomposition_dtype,
            )
            timing["filter"] += time.time() - currtime

            currtime = time.time()
            features += band_features
            timing["sum"] += time.time() - currtime
        else:
            features.append(band_eigenvectors)

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

    if filter is None:
        eigenvalues = np.array(eigenvalues, dtype=decomposition_dtype)
        features = np.concatenate(features, axis=1, dtype=decomposition_dtype)
        return eigenvalues, features
    else:
        return features


def compute_hks(
    mesh: Mesh,
    max_eigenvalue: float = 1e-8,
    t_max: Optional[float] = None,
    t_min: Optional[float] = None,
    n_components: int = 32,
    band_size: int = 50,
    truncate_extra: bool = False,
    drop_first: bool = False,
    robust: bool = True,
    mollify_factor: float = 1e-5,
    decomposition_dtype: Optional[np.dtype] = np.float64,
    verbose: int = False,
):
    filter_func = get_hks_filter(t_max, t_min, n_components, dtype=decomposition_dtype)
    out = spectral_geometry_filter(
        mesh,
        filter_func,
        max_eigenvalue=max_eigenvalue,
        band_size=band_size,
        truncate_extra=truncate_extra,
        drop_first=drop_first,
        robust=robust,
        mollify_factor=mollify_factor,
        decomposition_dtype=decomposition_dtype,
        verbose=verbose,
    )
    return out


def compute_geometry_vectors(
    mesh: Mesh,
    max_eigenvalue: float = 1e-8,
    n_components: int = 32,
    band_size: int = 50,
    truncate_extra: bool = False,
    drop_first: bool = False,
    robust: bool = True,
    mollify_factor: float = 1e-5,
    decomposition_dtype: Optional[np.dtype] = np.float64,
    verbose: int = False,
):
    filter_func = construct_bspline_filter(0.0, max_eigenvalue, n_components)
    out = spectral_geometry_filter(
        mesh,
        filter_func,
        max_eigenvalue=max_eigenvalue,
        band_size=band_size,
        truncate_extra=truncate_extra,
        drop_first=drop_first,
        robust=robust,
        mollify_factor=mollify_factor,
        decomposition_dtype=decomposition_dtype,
        verbose=verbose,
    )
    return out
