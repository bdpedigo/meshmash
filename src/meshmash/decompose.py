import time
from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse as sparse
from robust_laplacian import point_cloud_laplacian
from scipy.interpolate import BSpline
from scipy.linalg import eigh
from scipy.sparse import coo_array, csc_array, csr_array, sparray
from tqdm.auto import tqdm

from .laplacian import cotangent_laplacian
from .types import Mesh, interpret_mesh


def decompose_laplacian(
    L: sparray,
    M: sparray,
    n_components: int = 100,
    op_inv: Optional[sparse.linalg.LinearOperator] = None,
    sigma: float = -1e-10,
    tol: float = 1e-10,
    ncv: Optional[int] = None,
    prefactor: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the generalised eigenvalue problem for a mesh Laplacian.

    Computes the ``n_components`` smallest-magnitude eigenpairs of
    :math:`L \\phi = \\lambda M \\phi` using ARPACK shift-invert mode.
    For small matrices the dense solver [eigh][scipy.linalg.eigh] is used
    instead.

    Parameters
    ----------
    L :
        Sparse cotangent-weight Laplacian matrix, shape ``(V, V)``.
    M :
        Sparse diagonal mass (area) matrix, shape ``(V, V)``.
    n_components :
        Number of smallest eigenvalues/eigenvectors to compute.
    op_inv :
        Pre-factored inverse operator to accelerate the ARPACK solve.  If
        ``None`` and ``prefactor`` is also ``None``, no pre-factorisation
        is performed.
    sigma :
        Shift applied in shift-invert mode.  A small negative value
        ensures the solver targets the smallest non-negative eigenvalues.
    tol :
        Convergence tolerance passed to [eigsh][scipy.sparse.linalg.eigsh].
    ncv :
        Number of Lanczos vectors.  ``None`` lets ARPACK choose.
    prefactor :
        Pre-factorisation strategy.  Currently only ``'lu'`` (sparse LU
        via [splu][scipy.sparse.linalg.splu]) is supported.

    Returns
    -------
    eigenvalues :
        Array of eigenvalues sorted in ascending order, shape
        ``(n_components,)``.
    eigenvectors :
        Array of corresponding eigenvectors, shape ``(V, n_components)``.
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
    n_components: int = 100,
    op_inv: Optional[sparse.linalg.LinearOperator] = None,
    sigma: float = -1e-10,
    tol: float = 1e-10,
    robust: bool = True,
    mollify_factor: float = 1e-5,
    prefactor: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Laplacian eigendecomposition of a mesh.

    Builds the cotangent Laplacian and mass matrix with
    [cotangent_laplacian][meshmash.laplacian.cotangent_laplacian], then delegates to
    [decompose_laplacian][meshmash.decompose.decompose_laplacian].

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    n_components :
        Number of smallest eigenpairs to compute.
    op_inv :
        Pre-factored inverse operator; see [decompose_laplacian][meshmash.decompose.decompose_laplacian].
    sigma :
        Shift for ARPACK shift-invert mode.
    tol :
        Convergence tolerance.
    robust :
        If ``True``, use the robust Laplacian (see
        [cotangent_laplacian][meshmash.laplacian.cotangent_laplacian]).
    mollify_factor :
        Mollification factor for the robust Laplacian.
    prefactor :
        Pre-factorisation strategy; see [decompose_laplacian][meshmash.decompose.decompose_laplacian].

    Returns
    -------
    eigenvalues :
        Sorted eigenvalues, shape ``(n_components,)``.
    eigenvectors :
        Corresponding eigenvectors, shape ``(V, n_components)``.
    """
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
    L: sparray,
    M: sparray,
    max_eigenvalue: float = 1e-9,
    band_size: int = 50,
    truncate_extra: bool = True,
    verbose: Union[bool, int] = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenpairs of a mesh Laplacian up to a maximum eigenvalue
    using a band-by-band approach.

    Instead of computing all eigenpairs at once (which is slow for large
    meshes), this routine incrementally solves bands of ``band_size``
    eigenpairs, advancing the ARPACK shift to stay near the frontier of
    already-computed eigenvalues.  The result is equivalent to calling
    [decompose_laplacian][meshmash.decompose.decompose_laplacian] with a sufficiently large ``n_components``
    but uses less memory and is faster in practice.

    Parameters
    ----------
    L :
        Sparse cotangent-weight Laplacian matrix, shape ``(V, V)``.
    M :
        Sparse diagonal mass (area) matrix, shape ``(V, V)``.
    max_eigenvalue :
        Stop once the largest computed eigenvalue exceeds this value.
    band_size :
        Number of eigenpairs computed per ARPACK call.  Does not affect
        the result, only performance.
    truncate_extra :
        If ``True``, discard any eigenpairs whose eigenvalue exceeds
        ``max_eigenvalue`` (the last band may overshoot by up to
        ``band_size`` pairs).
    verbose :
        Verbosity level.  ``0`` or ``False`` is silent; ``>=1`` shows a
        progress bar; ``>=2`` also prints per-band diagnostics.

    Returns
    -------
    eigenvalues :
        Sorted eigenvalues, shape ``(K,)`` where ``K`` depends on
        ``max_eigenvalue``.
    eigenvectors :
        Corresponding eigenvectors, shape ``(V, K)``.

    Notes
    -----
    The band-shifting heuristic follows Section 4.1 of [1].

    References
    ----------
    [1] B. Vallet and B. Levy, "Spectral Geometry Processing with Manifold
        Harmonics", Computer Graphics Forum, 27(2):251-260, 2008.
    """
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
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a heat-kernel spectral filter for use with [spectral_geometry_filter][meshmash.decompose.spectral_geometry_filter].

    Returns a callable that converts an array of Laplacian eigenvalues into
    a 2-D array of HKS filter coefficients.

    Parameters
    ----------
    t_max :
        Largest diffusion timescale.
    t_min :
        Smallest diffusion timescale.
    n_scales :
        Number of timescales (= number of output HKS features).  Scales
        are spaced logarithmically between ``t_min`` and ``t_max``.
    dtype :
        Floating-point dtype for the output coefficients.

    Returns
    -------
    :
        Callable that accepts a 1-D eigenvalue array of length ``K`` and
        returns a ``(n_scales, K)`` coefficient array.
    """
    scales = np.geomspace(t_min, t_max, n_scales, dtype=dtype)

    def hks_filter(eigenvalues):
        coefs = np.exp(-np.outer(scales, eigenvalues))
        return coefs

    return hks_filter


def construct_bspline_basis(
    e_min: float, e_max: float, n_components: int
) -> list[BSpline]:
    """Construct a set of B-spline basis functions spanning an eigenvalue range.

    Parameters
    ----------
    e_min :
        Left boundary of the eigenvalue domain.
    e_max :
        Right boundary of the eigenvalue domain.
    n_components :
        Number of basis functions (= number of output geometry-vector
        features).

    Returns
    -------
    :
        List of ``n_components`` cubic [BSpline][scipy.interpolate.BSpline]
        basis elements, each non-zero over a compact sub-interval of
        ``[e_min, e_max]``.
    """
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


def construct_bspline_filter(
    e_min: float, e_max: float, n_components: int
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a B-spline spectral filter for use with [spectral_geometry_filter][meshmash.decompose.spectral_geometry_filter].

    Creates a bank of ``n_components`` cubic B-spline basis functions
    covering ``[e_min, e_max]`` via [construct_bspline_basis][meshmash.decompose.construct_bspline_basis] and
    wraps them in a callable that converts eigenvalues to filter
    coefficients.

    Parameters
    ----------
    e_min :
        Left boundary of the eigenvalue domain.
    e_max :
        Right boundary of the eigenvalue domain.
    n_components :
        Number of basis functions (= number of output features).

    Returns
    -------
    :
        Callable that accepts a 1-D eigenvalue array of length ``K`` and
        returns a ``(n_components, K)`` coefficient array.  Values outside
        the support of each basis element are set to ``0``.
    """
    bases = construct_bspline_basis(e_min, e_max, n_components)

    def bspline_filter(eigenvalues):
        coefs = np.stack([b(eigenvalues) for b in bases])
        coefs[~np.isfinite(coefs)] = 0
        return coefs

    return bspline_filter


def spectral_geometry_filter(
    mesh: Mesh,
    filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    max_eigenvalue: float = 1e-8,
    band_size: int = 50,
    truncate_extra: bool = True,
    drop_first: bool = True,
    decomposition_dtype: Optional[np.dtype] = np.float64,
    robust: bool = True,
    mollify_factor: float = 1e-5,
    point_laplacian: bool = False,
    n_neighbors: int = 30,
    verbose: Union[bool, int] = False,
    signals: Optional[np.ndarray] = None,
    signal_dtype: np.dtype = np.float64,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
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
    signals :
        Optional per-vertex signals to filter, shape ``(V, S)``. Where the
        default path filters the *diagonal* of the heat kernel -- one scalar
        per vertex per filter -- this filters the kernel's action on a
        function, :math:`(K_t f)(x) = \\sum_k c_t(\\lambda_k) \\phi_k(x)
        \\langle \\phi_k, f \\rangle_M`, accumulated band by band beside the
        diagonal off the same eigenpairs. Requires ``filter``.
    signal_dtype :
        Dtype the signal accumulation runs in, independent of
        ``decomposition_dtype``. Defaults to float64: a caller forming second
        moments differences a large number from a nearly equal one, and
        float32 does not carry that.

    Returns
    -------
    :
        Three shapes, by what was asked for. With ``filter`` and no
        ``signals``, a ``(V, F)`` array of filtered diagonal features. With
        ``filter`` and ``signals``, the pair ``(features, signal_features)``
        where ``signal_features`` is ``(V, F, S)``. With no ``filter``, the
        pair ``(eigenvalues, eigenvectors)`` -- the decomposition itself,
        unfiltered.

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
    if isinstance(mesh, tuple) and isinstance(
        mesh[0], (csr_array, csc_array, coo_array)
    ):
        L, M = mesh
    else:
        if not point_laplacian:
            L, M = cotangent_laplacian(
                mesh, robust=robust, mollify_factor=mollify_factor
            )
        else:
            vertices, _ = mesh
            L, M = point_cloud_laplacian(
                vertices,
                n_neighbors=n_neighbors,
                mollify_factor=mollify_factor,
            )
            # _, M = cotangent_laplacian(
            #     mesh, robust=robust, mollify_factor=mollify_factor
            # )

    if signals is not None:
        if filter is None:
            raise ValueError(
                "signals need a filter: without one this function returns the "
                "eigenpairs themselves and there is nothing to weight the "
                "signal projections by"
            )
        signals = np.asarray(signals, dtype=signal_dtype)
        if signals.ndim != 2 or len(signals) != L.shape[0]:
            raise ValueError(
                f"signals must be (V, S) with V={L.shape[0]}, got {signals.shape}"
            )

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

    if signals is not None:
        # M-weighted once, outside the loop: every band projects the signals
        # against the same <., .>_M inner product.
        weighted_signals = np.asarray(M @ signals, dtype=signal_dtype)
        signal_features = np.zeros(
            (L.shape[0], n_features, signals.shape[1]), dtype=signal_dtype
        )
    else:
        signal_features = None

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

            if signals is not None:
                currtime = time.time()
                band_phi = band_eigenvectors[:, first_idx:].astype(signal_dtype)
                # <phi_k, f>_M, for every kept eigenvector and every signal.
                projected = band_phi.T @ weighted_signals
                # One GEMM rather than a loop over filters: fold the (F, K)
                # coefficients into the (K, S) projections to get (K, F * S),
                # then left-multiply by the eigenvectors once.
                folded = (
                    np.asarray(band_coefs, dtype=signal_dtype)[:, :, None]
                    * projected[None, :, :]
                ).transpose(1, 0, 2)
                signal_features += (
                    band_phi @ folded.reshape(band_phi.shape[1], -1)
                ).reshape(signal_features.shape)
                timing["signals"] = timing.get("signals", 0) + (
                    time.time() - currtime
                )
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
    elif signals is not None:
        return features, signal_features
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
    point_laplacian: bool = False,
    n_neighbors: int = 30,
    boundary_aware: bool = False,
    boundary_weight: float = 0.5,
    boundary_indices: Optional[np.ndarray] = None,
    verbose: Union[bool, int] = False,
) -> np.ndarray:
    """Compute the Heat Kernel Signature (HKS) for each vertex of a mesh.

    The HKS is a multi-scale, intrinsic shape descriptor based on the
    diagonal of the heat kernel at a set of diffusion timescales.  It
    captures local geometry from fine detail (small ``t``) to global
    structure (large ``t``).

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    max_eigenvalue :
        Maximum Laplacian eigenvalue to include in the computation.
        Larger values capture finer geometric detail at increased cost.
    t_min :
        Smallest diffusion timescale.
    t_max :
        Largest diffusion timescale.
    n_components :
        Number of diffusion timescales.  Scales are spaced logarithmically
        between ``t_min`` and ``t_max``, and each scale produces one
        feature per vertex.
    band_size :
        Number of eigenpairs per ARPACK band; see
        [spectral_geometry_filter][meshmash.decompose.spectral_geometry_filter].
    truncate_extra :
        Whether to discard eigenpairs that overshoot ``max_eigenvalue``.
    drop_first :
        If ``True``, drop the first (near-zero) eigenpair before applying
        the filter.  The first eigenvector is proportional to vertex areas
        and is typically uninformative.
    robust :
        If ``True``, use the robust Laplacian (see
        [cotangent_laplacian][meshmash.laplacian.cotangent_laplacian]).
    mollify_factor :
        Mollification factor for the robust Laplacian.
    decomposition_dtype :
        Floating-point dtype for the eigendecomposition.
    point_laplacian :
        If ``True``, build a point-cloud Laplacian instead of the mesh
        cotangent Laplacian (useful for point clouds without face data).
    n_neighbors :
        Number of neighbours used when ``point_laplacian=True``.
    boundary_aware :
        If ``True``, treat open mesh boundaries as partially absorbing: the
        returned HKS is a convex mix of the Neumann (reflecting, standard)
        signature and a Dirichlet (absorbing) signature computed on the
        interior vertices only.  For a closed mesh this is a no-op.
    boundary_weight :
        Mixing weight for the Dirichlet signature when ``boundary_aware`` is
        set: ``(1 - w) * neumann + w * dirichlet``.  ``0.5`` means boundary
        vertices reflect half the heat and let the other half exit.
    boundary_indices :
        Optional precomputed vertex indices lying on open boundaries.  If
        ``None`` and ``boundary_aware`` is set, they are detected from the
        mesh faces via
        [get_submesh_borders][meshmash.split.get_submesh_borders].
    verbose :
        Verbosity level passed to [spectral_geometry_filter][meshmash.decompose.spectral_geometry_filter].

    Returns
    -------
    :
        Per-vertex HKS feature array of shape ``(V, n_components)``.

    Notes
    -----
    The band-by-band eigensolver from [1] is used for efficiency.  The
    robust Laplacian from [2] is recommended for real-world meshes.

    References
    ----------
    [1] J. Sun, M. Ovsjanikov, and L. Guibas, "A Concise and Provably
        Informative Multi-Scale Signature Based on Heat Diffusion",
        Computer Graphics Forum, 28(5):1383-1392, 2009.
    [2] N. Sharp and K. Crane, "A Laplacian for Nonmanifold Triangle
        Meshes", Computer Graphics Forum, 39(5), 2020.
    """
    filter_func = get_hks_filter(t_max, t_min, n_components, dtype=decomposition_dtype)
    if boundary_aware:
        return _compute_boundary_aware_hks(
            mesh,
            filter_func,
            max_eigenvalue=max_eigenvalue,
            band_size=band_size,
            truncate_extra=truncate_extra,
            drop_first=drop_first,
            robust=robust,
            mollify_factor=mollify_factor,
            decomposition_dtype=decomposition_dtype,
            boundary_weight=boundary_weight,
            boundary_indices=boundary_indices,
            verbose=verbose,
        )
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
        point_laplacian=point_laplacian,
        n_neighbors=n_neighbors,
        verbose=verbose,
    )
    return out


def _compute_boundary_aware_hks(
    mesh: Mesh,
    filter_func: Callable[[np.ndarray], np.ndarray],
    max_eigenvalue: float,
    band_size: int,
    truncate_extra: bool,
    drop_first: bool,
    robust: bool,
    mollify_factor: float,
    decomposition_dtype: Optional[np.dtype],
    boundary_weight: float,
    boundary_indices: Optional[np.ndarray],
    verbose: Union[bool, int],
) -> np.ndarray:
    """HKS that mixes a reflecting (Neumann) and absorbing (Dirichlet) signature.

    The standard cotangent/robust Laplacian on a mesh with boundary already
    imposes natural (Neumann) boundary conditions, so its HKS is the
    reflecting signature.  The Dirichlet signature is obtained by solving the
    same generalized eigenproblem on the interior rows/columns only, then
    scattering the interior features back with zeros on the boundary.  On the
    boundary the Dirichlet contribution vanishes, so the mix leaves boundary
    vertices at ``(1 - boundary_weight)`` of their reflecting value.
    """
    if isinstance(mesh, tuple) and isinstance(
        mesh[0], (csr_array, csc_array, coo_array)
    ):
        raise ValueError(
            "boundary_aware HKS requires a (vertices, faces) mesh so boundaries "
            "can be detected; pass boundary_indices explicitly to use an (L, M) mesh."
        )

    L, M = cotangent_laplacian(mesh, robust=robust, mollify_factor=mollify_factor)

    neumann = spectral_geometry_filter(
        (L, M),
        filter_func,
        max_eigenvalue=max_eigenvalue,
        band_size=band_size,
        truncate_extra=truncate_extra,
        drop_first=drop_first,
        decomposition_dtype=decomposition_dtype,
        verbose=verbose,
    )

    if boundary_indices is None:
        from .split import get_submesh_borders

        boundary_indices = get_submesh_borders(mesh)

    # Closed mesh (no open boundary): nothing to absorb.
    if len(boundary_indices) == 0:
        return neumann

    n = L.shape[0]
    interior = np.ones(n, dtype=bool)
    interior[boundary_indices] = False
    interior_idx = np.flatnonzero(interior)

    L_II = L.tocsr()[interior_idx][:, interior_idx]
    # M is a diagonal (lumped-area) matrix; restrict its diagonal to the interior.
    M_II = sparse.dia_array(
        (np.asarray(M.diagonal())[interior_idx], 0),
        shape=(interior_idx.size, interior_idx.size),
    )

    # Interior Laplacian is positive definite (no constant null mode), so the
    # first eigenpair must NOT be dropped here.
    dirichlet_interior = spectral_geometry_filter(
        (L_II, M_II),
        filter_func,
        max_eigenvalue=max_eigenvalue,
        band_size=band_size,
        truncate_extra=truncate_extra,
        drop_first=False,
        decomposition_dtype=decomposition_dtype,
        verbose=verbose,
    )

    dirichlet = np.zeros_like(neumann)
    dirichlet[interior_idx] = dirichlet_interior

    return (1.0 - boundary_weight) * neumann + boundary_weight * dirichlet


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
    verbose: Union[bool, int] = False,
) -> np.ndarray:
    """Compute spectral geometry descriptors using a B-spline spectral filter.

    Similar in spirit to [compute_hks][meshmash.decompose.compute_hks], but instead of heat-kernel
    exponentials the spectrum is partitioned by a bank of cubic B-spline
    basis functions.  Each basis function acts as a band-pass filter,
    yielding one feature per vertex per band.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    max_eigenvalue :
        Maximum Laplacian eigenvalue to include.  Determines the upper
        boundary of the B-spline domain.
    n_components :
        Number of B-spline basis functions (= number of output features
        per vertex).
    band_size :
        Number of eigenpairs per ARPACK band; see
        [spectral_geometry_filter][meshmash.decompose.spectral_geometry_filter].
    truncate_extra :
        Whether to discard eigenpairs that overshoot ``max_eigenvalue``.
    drop_first :
        If ``True``, drop the first (near-zero) eigenpair.
    robust :
        If ``True``, use the robust Laplacian (see
        [cotangent_laplacian][meshmash.laplacian.cotangent_laplacian]).
    mollify_factor :
        Mollification factor for the robust Laplacian.
    decomposition_dtype :
        Floating-point dtype for the eigendecomposition.
    verbose :
        Verbosity level passed to [spectral_geometry_filter][meshmash.decompose.spectral_geometry_filter].

    Returns
    -------
    :
        Per-vertex geometry-vector feature array of shape
        ``(V, n_components)``.

    Notes
    -----
    The B-spline filter bank follows [1].

    References
    ----------
    [1] R. Litman and A. M. Bronstein, "Learning spectral descriptors for
        deformable shape correspondence", IEEE Transactions on Pattern
        Analysis and Machine Intelligence, 36(1):171-180, 2013.
    """
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


#: The invariants [compute_heat_kernel_moments][meshmash.decompose.compute_heat_kernel_moments]
#: emits per timescale, in the order it emits them.
HEAT_KERNEL_MOMENT_NAMES = (
    "drift",
    "normal",
    "tangent",
    "extent",
    "linear",
    "planar",
    "round",
    "align1",
    "align2",
)

#: The upper triangle of a symmetric 3x3, in the order the second-moment
#: signals are built and read back.
_MOMENT_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def heat_kernel_moment_names(n_scales: int) -> list[str]:
    """Column names for a [compute_heat_kernel_moments][meshmash.decompose.compute_heat_kernel_moments] result.

    Scale-major, matching the array's ``(V, n_scales, 8)`` layout before it is
    flattened: every invariant of scale 0, then every invariant of scale 1.

    Parameters
    ----------
    n_scales :
        Number of timescales the moments were computed over.

    Returns
    -------
    :
        ``n_scales * 8`` names of the form ``drift_0``, ``normal_0``, ....
    """
    return [
        f"{name}_{index}"
        for index in range(n_scales)
        for name in HEAT_KERNEL_MOMENT_NAMES
    ]


def vertex_normals(mesh: Mesh) -> np.ndarray:
    """Unit vertex normals, area-weighted from the face normals.

    The sign is whatever the mesh's face winding says, which for a mesh nobody
    has oriented is arbitrary per connected component. Callers that need a
    consistent sign have to fix it themselves — see
    [compute_heat_kernel_moments][meshmash.decompose.compute_heat_kernel_moments],
    which fixes it against the mean-curvature direction.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].

    Returns
    -------
    :
        Unit normals of shape ``(V, 3)``. A vertex touching no face, or whose
        face normals cancel exactly, comes back as the zero vector.
    """
    vertices, faces = interpret_mesh(mesh)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)

    corners = vertices[faces]
    # |cross| is twice the triangle area, so summing the raw cross products
    # area-weights the average for free.
    crossed = np.cross(
        corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    )

    flat = faces.reshape(-1)
    repeated = np.repeat(crossed, 3, axis=0)
    normals = np.empty((len(vertices), 3), dtype=np.float64)
    for axis in range(3):
        normals[:, axis] = np.bincount(
            flat, weights=repeated[:, axis], minlength=len(vertices)
        )

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(lengths > 0, lengths, 1.0)


def compute_heat_kernel_moments(
    mesh: Mesh,
    max_eigenvalue: float = 1e-8,
    t_max: Optional[float] = None,
    t_min: Optional[float] = None,
    n_scales: int = 8,
    band_size: int = 50,
    truncate_extra: bool = True,
    robust: bool = True,
    mollify_factor: float = 1e-5,
    decomposition_dtype: Optional[np.dtype] = np.float64,
    moment_dtype: np.dtype = np.float64,
    out_dtype: np.dtype = np.float32,
    verbose: Union[bool, int] = False,
) -> np.ndarray:
    """Rotation-invariant spatial moments of the local heat kernel, per vertex.

    Where the heat kernel signature reads the kernel's *diagonal* —
    :math:`k_t(x, x)`, how much heat stays put — this reads the kernel's first
    and second spatial moments: where the heat went, and how the cloud it
    spread into is shaped.  Treating :math:`k_t(x, \\cdot)` as a probability
    measure on the surface,

    - the **drift** :math:`m_t(x) = \\int k_t(x, y)\\, y\\, dA - x` is the
      heat-smoothed position minus the original one.  At small ``t`` it is the
      mean-curvature normal, :math:`m_t \\approx t H \\mathbf{n}`; at larger
      ``t`` it is a one-sidedness detector, near zero wherever heat can leave
      symmetrically and large wherever it cannot.
    - the **covariance** :math:`C_t(x)` is an intrinsic multiscale local PCA.
      Diffusion weights follow the surface rather than a Euclidean ball, so
      the neighbourhood does not leak across a gap that is close in space but
      far along the mesh.

    Both are computed by spectral filtering of nine signals — the three
    coordinate functions and their six pairwise products — off the same
    eigenpairs the HKS uses, via
    [spectral_geometry_filter][meshmash.decompose.spectral_geometry_filter]'s
    ``signals`` argument.  The measure needs no normalising: eigenvectors are
    M-orthonormal and only the constant mode has nonzero
    :math:`\\langle \\phi_k, 1 \\rangle_M`, so :math:`\\int k_t(x, y)\\, dA = 1`
    exactly however far the spectrum is truncated.  **The constant mode is
    therefore never dropped** — it is what carries the local mean — which is
    the one place this differs from
    [compute_hks][meshmash.decompose.compute_hks], where dropping it removes
    only an additive constant.

    Eight invariants come back per timescale, in
    ``HEAT_KERNEL_MOMENT_NAMES`` order, with
    :math:`\\lambda_1 \\ge \\lambda_2 \\ge \\lambda_3` the eigenvalues of
    :math:`C_t`:

    - ``drift``: :math:`\\lVert m_t \\rVert`, in mesh length units.
    - ``normal``: :math:`m_t \\cdot \\mathbf{n}`, signed — the drift pushed
      off the surface rather than along it.
    - ``tangent``: :math:`\\lVert m_t - (m_t \\cdot \\mathbf{n})\\mathbf{n}
      \\rVert`, the drift pushed *along* the surface — the one-sidedness
      detector, and not recoverable from the two columns above by a tree
      model, which cannot take the square root of a difference. On a tube it
      is near zero however curved the tube is, because heat leaves equally in
      both directions; it rises wherever one direction is closed off. Without
      it ``drift`` reads a dendrite-radius tube and a spine cap as nearly the
      same thing, because a tube's own mean curvature dominates the drift's
      magnitude.
    - ``extent``: :math:`\\sqrt{\\operatorname{tr} C_t}`, the neighbourhood's
      overall size in length units.
    - ``linear``, ``planar``, ``round``: :math:`(\\lambda_1 -
      \\lambda_2)/\\lambda_1`, :math:`(\\lambda_2 - \\lambda_3)/\\lambda_1`,
      :math:`\\lambda_3/\\lambda_1`.
    - ``align1``, ``align2``: :math:`m_t' C_t m_t` and :math:`m_t' C_t^2 m_t`,
      each normalised by :math:`\\lVert m_t \\rVert^2` and the matching power
      of :math:`\\lambda_1` — how much of the drift lies along the
      neighbourhood's dominant axis.

    ``extent`` with ``linear`` and ``planar`` is the eigenvalue triple
    re-expressed, not a subset of it: the map between them is a bijection.
    Ratios are emitted rather than raw eigenvalues because the consumers are
    tree models, which cannot form :math:`\\lambda_1 / \\lambda_3` themselves.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    max_eigenvalue :
        Maximum Laplacian eigenvalue to include; see
        [compute_hks][meshmash.decompose.compute_hks].
    t_max :
        Largest diffusion timescale.
    t_min :
        Smallest diffusion timescale.
    n_scales :
        Number of timescales, spaced logarithmically between ``t_min`` and
        ``t_max``.  Each contributes eight columns.
    band_size :
        Number of eigenpairs per ARPACK band.
    truncate_extra :
        Whether to discard eigenpairs that overshoot ``max_eigenvalue``.
    robust :
        If ``True``, use the robust Laplacian.
    mollify_factor :
        Mollification factor for the robust Laplacian.
    decomposition_dtype :
        Floating-point dtype for the eigendecomposition.
    moment_dtype :
        Dtype the moments are accumulated and assembled in.  float64 is not
        a default worth changing: :math:`C_t` is the difference between a
        second moment scaled by the mesh's own extent and a nearly equal
        outer product, and at spine scale on a micron-sized chunk those
        differ by four orders of magnitude.
    out_dtype :
        Dtype of the returned invariants.  The invariants are well
        conditioned once formed, so float32 is enough for them even though
        it is not enough to form them.
    verbose :
        Verbosity level passed through to
        [spectral_geometry_filter][meshmash.decompose.spectral_geometry_filter].

    Returns
    -------
    :
        Array of shape ``(V, n_scales * 8)``, scale-major, named by
        [heat_kernel_moment_names][meshmash.decompose.heat_kernel_moment_names].

    Notes
    -----
    The invariants are extrinsic: they read the shape of the diffusion cloud
    in space, so a tube that curves appreciably within one diffusion length
    loses ``linear`` gradually.  That is a smooth degradation with scale, not
    a failure.

    Precedent for the construction: integral invariants [1], local covariance
    features from point-cloud processing [2], vector diffusion maps [3], and
    the classical identity between heat smoothing and mean-curvature flow.

    References
    ----------
    [1] H. Pottmann, J. Wallner, Q.-X. Huang, and Y.-L. Yang, "Integral
        invariants for robust geometry processing", Computer Aided Geometric
        Design, 26(1):37-60, 2009.
    [2] M. Weinmann, B. Jutzi, S. Hinz, and C. Mallet, "Semantic point cloud
        interpretation based on optimal neighborhoods, relevant features and
        efficient classifiers", ISPRS Journal of Photogrammetry and Remote
        Sensing, 105:286-304, 2015.
    [3] A. Singer and H.-T. Wu, "Vector diffusion maps and the connection
        Laplacian", Communications on Pure and Applied Mathematics,
        65(8):1067-1144, 2012.
    """
    vertices, faces = interpret_mesh(mesh)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)

    L, M = cotangent_laplacian(
        (vertices, faces), robust=robust, mollify_factor=mollify_factor
    )

    # Centred on the mesh's own centroid before the products are formed. The
    # coordinates arrive as absolute dataset positions, which can be six
    # orders of magnitude larger than the local structure being measured, and
    # every digit of that offset is a digit the covariance's cancellation
    # would eat.
    centered = vertices - vertices.mean(axis=0)
    signals = np.empty((len(vertices), 9), dtype=moment_dtype)
    signals[:, :3] = centered
    for column, (i, j) in enumerate(_MOMENT_PAIRS, start=3):
        signals[:, column] = centered[:, i] * centered[:, j]

    filter_func = get_hks_filter(t_max, t_min, n_scales, dtype=decomposition_dtype)
    _, filtered = spectral_geometry_filter(
        (L, M),
        filter_func,
        max_eigenvalue=max_eigenvalue,
        band_size=band_size,
        truncate_extra=truncate_extra,
        drop_first=False,
        decomposition_dtype=decomposition_dtype,
        signals=signals,
        signal_dtype=moment_dtype,
        verbose=verbose,
    )

    mean = filtered[:, :, :3]
    second = filtered[:, :, 3:]
    drift = mean - centered[:, None, :].astype(moment_dtype)

    covariance = np.empty(mean.shape + (3,), dtype=moment_dtype)
    for column, (i, j) in enumerate(_MOMENT_PAIRS):
        entry = second[:, :, column] - mean[:, :, i] * mean[:, :, j]
        covariance[:, :, i, j] = entry
        covariance[:, :, j, i] = entry

    # Descending, and clipped: a covariance is positive semidefinite in exact
    # arithmetic, and the smallest eigenvalue of a nearly degenerate one comes
    # back slightly negative from the cancellation above.
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance)[:, :, ::-1], 0.0, None)
    largest = eigenvalues[:, :, 0]
    positive = largest > 0
    safe = np.where(positive, largest, 1.0)

    drift_norm = np.linalg.norm(drift, axis=-1)
    covariance_drift = np.einsum("vtij,vtj->vti", covariance, drift)
    drifting = drift_norm > 0
    scaled = np.where(drifting & positive, drift_norm**2, 1.0)

    normals = vertex_normals((vertices, faces))
    # The winding of a mesh nobody oriented is arbitrary, so the absolute sign
    # of a vertex normal says nothing. What is not arbitrary is the smallest
    # scale's drift: heat smoothing is mean-curvature flow, so it points
    # toward the centre of curvature whichever way the faces wind. Flipping
    # the normals into anti-alignment with it fixes the outward convention
    # from the geometry rather than from the file.
    reference = drift[:, 0, :]
    reference_norm = np.linalg.norm(reference, axis=-1)
    usable = reference_norm > 0
    if usable.any():
        cosines = (
            reference[usable] / reference_norm[usable, None] * normals[usable]
        ).sum(axis=-1)
        if np.median(cosines) > 0:
            normals = -normals

    normal_drift = np.einsum("vti,vi->vt", drift, normals.astype(moment_dtype))
    # Pythagoras rather than subtracting the vector and re-normalising: the
    # two give the same number and this is a third of the arithmetic. The clip
    # is for the roundoff where the drift is almost entirely normal.
    tangent_drift = np.sqrt(np.clip(drift_norm**2 - normal_drift**2, 0.0, None))

    invariants = np.stack(
        [
            drift_norm,
            normal_drift,
            tangent_drift,
            np.sqrt(eigenvalues.sum(axis=-1)),
            np.where(positive, (eigenvalues[..., 0] - eigenvalues[..., 1]) / safe, 0.0),
            np.where(positive, (eigenvalues[..., 1] - eigenvalues[..., 2]) / safe, 0.0),
            np.where(positive, eigenvalues[..., 2] / safe, 0.0),
            np.where(
                drifting & positive,
                (drift * covariance_drift).sum(axis=-1) / (scaled * safe),
                0.0,
            ),
            np.where(
                drifting & positive,
                (covariance_drift * covariance_drift).sum(axis=-1)
                / (scaled * safe**2),
                0.0,
            ),
        ],
        axis=-1,
    )
    return invariants.reshape(len(vertices), -1).astype(out_dtype)
