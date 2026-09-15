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
from .types import ArrayLike, Mesh


def arpack_start_vector(n: int, seed: Optional[int]) -> Optional[np.ndarray]:
    """The starting residual vector for an ARPACK solve, drawn reproducibly.

    ARPACK draws its own when none is given, and that is the whole source of
    run-to-run variation in every spectral routine here.  Its generator lives
    in Fortran ``SAVE`` state, so the seed is not per call: the first solve in
    a process gets one vector and the second gets another, and two identical
    calls in one session disagree at the decomposition dtype.  Near a
    degenerate eigenvalue, or at the band truncation edge, that is enough to
    return a different basis or a different mode count.

    The distribution here is ARPACK's own — ``dgetv0`` fills the vector with
    ``dlarnv(idist=2)``, which is uniform on ``(-1, 1)`` — so seeding changes
    which vector is drawn and not the kind of vector.  Any vector with a
    component along the wanted invariant subspace converges to the same
    eigenpairs, so this is a reproducibility control and not an accuracy one.

    Parameters
    ----------
    n :
        Length of the vector, which is the matrix dimension.
    seed :
        Seed for [default_rng][numpy.random.default_rng].  ``None`` returns
        ``None``, which leaves ARPACK to draw its own as before.

    Returns
    -------
    :
        A vector of shape ``(n,)`` uniform on ``(-1, 1)``, or ``None``.
    """
    if seed is None:
        return None
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=n)


def decompose_laplacian(
    L: sparray,
    M: sparray,
    n_components: int = 100,
    op_inv: Optional[sparse.linalg.LinearOperator] = None,
    sigma: float = -1e-10,
    tol: float = 1e-10,
    ncv: Optional[int] = None,
    prefactor: Optional[str] = None,
    profile: Optional[dict] = None,
    seed: Optional[int] = None,
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
    seed :
        Seed for the ARPACK starting vector.  ``None`` lets ARPACK draw its
        own, which is why two runs of the same call do not agree bit for bit:
        ARPACK's generator carries state across calls within a process, so the
        second call in a process starts somewhere else.  An integer draws the
        vector here instead, from the same distribution ARPACK uses, and makes
        the result reproducible.  See Notes.
    ncv :
        Number of Lanczos vectors.  ``None`` lets ARPACK choose.
    prefactor :
        Pre-factorisation strategy.  Currently only ``'lu'`` (sparse LU
        via [splu][scipy.sparse.linalg.splu]) is supported.
    profile :
        Optional dict that accumulates solver cost: seconds under
        ``"factor"`` (the shift-invert LU) and ``"arpack"`` (the Lanczos
        iteration), or ``"dense"`` for the small-matrix path. Passing it
        does not change the result: the same LU ``eigsh`` builds
        internally is built here so the two phases time apart.

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
        currtime = time.time()
        eigenvalues, eigenvectors = eigh(L.toarray(), M.toarray())
        if profile is not None:
            profile["dense"] = profile.get("dense", 0.0) + time.time() - currtime
    else:
        if profile is not None and op_inv is None:
            # The same splu eigsh would build internally, done here so the
            # factorization is timed apart from the Lanczos iteration.
            currtime = time.time()
            lu = sparse.linalg.splu((L - sigma * M).tocsc())
            op_inv = sparse.linalg.LinearOperator(
                matvec=lu.solve, shape=L.shape, dtype=L.dtype
            )
            profile["factor"] = profile.get("factor", 0.0) + time.time() - currtime
        currtime = time.time()
        eigenvalues, eigenvectors = sparse.linalg.eigsh(
            L,
            k=n_components,
            M=M,
            sigma=sigma,
            OPinv=op_inv,
            tol=tol,
            ncv=ncv,
            v0=arpack_start_vector(L.shape[0], seed),
        )
        if profile is not None:
            profile["arpack"] = profile.get("arpack", 0.0) + time.time() - currtime
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
    seed: Optional[int] = None,
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
    seed :
        Seed for the ARPACK starting vector, forwarded to
        [decompose_laplacian][meshmash.decompose.decompose_laplacian].  ``None``
        lets ARPACK draw its own, so two runs of the same call agree only to
        the decomposition dtype.  An integer makes the result reproducible.

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
            L, M, n_components=band_size, sigma=sigma, seed=seed
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
    return get_heat_filter(
        np.geomspace(t_min, t_max, n_scales, dtype=dtype), dtype=dtype
    )


def get_heat_filter(
    scales: ArrayLike, dtype: np.dtype = np.float64
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a heat-kernel spectral filter over explicitly chosen timescales.

    The same :math:`\\exp(-t \\lambda)` bank that
    [get_hks_filter][meshmash.decompose.get_hks_filter] builds, but from a
    timescale array the caller supplies rather than from a geometric grid.
    Useful when the timescales come from a length scale of interest, since a
    diffusion of timescale ``t`` smooths over roughly ``sqrt(t)`` of surface.

    Parameters
    ----------
    scales :
        Diffusion timescales, shape ``(T,)``.  Need not be evenly spaced.
    dtype :
        Floating-point dtype for the output coefficients.

    Returns
    -------
    :
        Callable that accepts a 1-D eigenvalue array of length ``K`` and
        returns a ``(T, K)`` coefficient array.
    """
    scales = np.asarray(scales, dtype=dtype)

    def heat_filter(eigenvalues):
        coefs = np.exp(-np.outer(scales, eigenvalues))
        return coefs

    return heat_filter


def concatenate_filters(
    *filters: Optional[Callable[[np.ndarray], np.ndarray]],
) -> Callable[[np.ndarray], np.ndarray]:
    """Stack several spectral filters into one bank, in the order given.

    One decomposition is the expensive part of
    [spectral_geometry_filter][meshmash.decompose.spectral_geometry_filter], and
    a stacked bank lets one call serve several filters off it.  The stacked
    filter's output rows are the inputs' output rows, concatenated, so a caller
    slices the result back apart by the widths it put in.

    ``None`` entries are skipped, so an optional filter can be passed straight
    through without a branch at the call site.

    Parameters
    ----------
    *filters :
        Filters as accepted by
        [spectral_geometry_filter][meshmash.decompose.spectral_geometry_filter],
        each mapping a ``(K,)`` eigenvalue array to a ``(F_i, K)`` coefficient
        array.

    Returns
    -------
    :
        Callable returning a ``(sum(F_i), K)`` coefficient array.

    Raises
    ------
    ValueError
        If every argument is ``None``.
    """
    kept = [f for f in filters if f is not None]
    if not kept:
        raise ValueError(
            "concatenate_filters needs at least one filter that is not None"
        )

    if len(kept) == 1:
        return kept[0]

    def concatenated_filter(eigenvalues):
        return np.concatenate([f(eigenvalues) for f in kept], axis=0)

    return concatenated_filter


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
    overlap_target: Optional[int] = None,
    profile: Optional[dict] = None,
    seed: Optional[int] = None,
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
    seed :
        Seed for the ARPACK starting vector, forwarded to
        [decompose_laplacian][meshmash.decompose.decompose_laplacian].  ``None``
        lets ARPACK draw its own, so two runs of the same call agree only to
        the decomposition dtype.  An integer makes the result reproducible.
    signals :
        Optional per-vertex signals to filter, shape ``(V, S)``. Where the
        default path filters the *diagonal* of the heat kernel -- one scalar
        per vertex per filter -- this filters the kernel's action on a
        function, :math:`(K_t f)(x) = \\sum_k c_t(\\lambda_k) \\phi_k(x)
        \\langle \\phi_k, f \\rangle_M`, accumulated band by band beside the
        diagonal off the same eigenpairs. Requires ``filter``.
    signal_dtype :
        Dtype the signal accumulation runs in, independent of
        ``decomposition_dtype``. Defaults to float64, which is a deliberate
        asymmetry: the decomposition tolerates float32, but a projection sums
        over every vertex of the mesh and a caller differencing two filtered
        signals needs the digits that float32 does not carry.
    overlap_target :
        ``None`` keeps the original placement heuristic (next shift 0.4
        bandwidths past the frontier), which re-solves roughly a quarter
        of every band at the seam. An integer switches to predicted
        placement: the eigenvalue density observed so far (flat for a 2-D
        surface, by Weyl's law) places the next shift so the seam overlap
        lands near this many pairs, and shrinks the final band to the
        predicted remainder instead of overshooting the cutoff by up to a
        whole band. The eigenpairs kept are the same; only the redundant
        solves shrink. Must be at least 1 — the band-continuity check
        needs a nonempty seam.
    profile :
        Optional dict that accumulates the band loop's cost breakdown and
        does not change the result. Seconds: ``"factor"``, ``"arpack"``
        (from [decompose_laplacian][meshmash.decompose.decompose_laplacian]),
        plus this loop's ``"decompose"``, ``"filter"``, ``"sum"``. Counters:
        ``"n_bands"``, ``"n_eigenpairs"``, and the redundancy the sigma
        heuristic pays — ``"n_retries"``/``"retry_pairs"`` (bands thrown
        away whole), ``"overlap_pairs"`` (re-solved at seams),
        ``"overshoot_pairs"`` (truncated past ``max_eigenvalue``).

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

    if overlap_target is not None and overlap_target < 1:
        raise ValueError(
            "overlap_target must be at least 1: the band-continuity check "
            "needs the new band to reach back over the frontier"
        )

    eigenvalues = []
    band_max_eigenvalue = 0
    sigma = -1e-10
    last_eigenvalue = 0
    eigenvalue_bandwidth = 0
    band_k = band_size
    place_next = False

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
        if place_next:
            # Predicted placement (overlap_target set): the density seen so
            # far — flat in eigenvalue for a 2-D surface, by Weyl's law —
            # converts pair counts to shifts. The next band of k pairs
            # centers on sigma, so putting sigma (k/2 - target) pairs past
            # the frontier lands the seam near `overlap_target` pairs.
            density = len(eigenvalues) / band_max_eigenvalue
            remaining = (max_eigenvalue - band_max_eigenvalue) * density
            margin = max(8.0, 0.25 * remaining)
            if remaining + overlap_target + margin < band_size:
                # Final band: solve the predicted remainder, not a full band.
                # Floor at 2*target+2 so sigma stays past the frontier.
                band_k = int(
                    max(
                        np.ceil(remaining + overlap_target + margin),
                        2 * overlap_target + 2,
                        8,
                    )
                )
            else:
                band_k = band_size
            sigma = band_max_eigenvalue + (0.5 * band_k - overlap_target) / density
            place_next = False

        if verbose >= 2:
            print(f"Computing band with sigma={sigma:.3g}")

        currtime = time.time()
        band_eigenvalues, band_eigenvectors = decompose_laplacian(
            L,
            M,
            n_components=band_k,
            sigma=sigma,
            tol=eigen_tol,
            profile=profile,
            seed=seed,
        )
        timing["decompose"] += time.time() - currtime

        # find the index where the new eigenvalues are within the tolerance
        # of the last eigenvalue
        diffs = np.abs(band_eigenvalues - last_eigenvalue)
        if (np.min(diffs)) > tol and (len(eigenvalues) > 0):  # ignore if 1st
            # retry with a smaller sigma
            sigma = sigma - 0.2 * eigenvalue_bandwidth
            if profile is not None:
                profile["n_retries"] = profile.get("n_retries", 0) + 1
                profile["retry_pairs"] = profile.get("retry_pairs", 0) + band_size
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
            if profile is not None:
                profile["overlap_pairs"] = (
                    profile.get("overlap_pairs", 0) + int(closest_idx) + 1
                )
            band_eigenvalues = band_eigenvalues[closest_idx + 1 :]
            band_eigenvectors = band_eigenvectors[:, closest_idx + 1 :]

        if truncate_extra and (band_eigenvalues[-1] > max_eigenvalue):
            # Truncate to the max_eigenvalue
            truncation_idx = np.searchsorted(band_eigenvalues, max_eigenvalue)
            if profile is not None:
                profile["overshoot_pairs"] = profile.get("overshoot_pairs", 0) + max(
                    len(band_eigenvalues) - int(truncation_idx) - 1, 0
                )
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
                timing["signals"] = timing.get("signals", 0) + (time.time() - currtime)
        else:
            features.append(band_eigenvectors)

        # update values for next iteration
        if profile is not None:
            profile["n_bands"] = profile.get("n_bands", 0) + 1
        eigenvalues.extend(band_eigenvalues)
        band_max_eigenvalue = np.max(band_eigenvalues)
        band_min_eigenvalue = np.min(band_eigenvalues)
        eigenvalue_bandwidth = band_max_eigenvalue - band_min_eigenvalue
        if overlap_target is None:
            sigma = band_max_eigenvalue + 0.4 * eigenvalue_bandwidth
        else:
            place_next = True

        # update by the amount the max eigenvalue increased
        pbar.update(band_max_eigenvalue - last_eigenvalue)
        last_eigenvalue = band_eigenvalues[-1]

    pbar.close()

    if profile is not None:
        profile["n_eigenpairs"] = profile.get("n_eigenpairs", 0) + len(eigenvalues)
        for key, value in timing.items():
            profile[key] = profile.get(key, 0.0) + value

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
    truncate_extra: bool = True,
    drop_first: bool = True,
    robust: bool = True,
    mollify_factor: float = 1e-5,
    decomposition_dtype: Optional[np.dtype] = np.float64,
    point_laplacian: bool = False,
    n_neighbors: int = 30,
    boundary_aware: bool = False,
    boundary_weight: float = 0.5,
    boundary_indices: Optional[np.ndarray] = None,
    verbose: Union[bool, int] = False,
    seed: Optional[int] = None,
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
        If ``True``, drop the constant eigenpair before applying the filter.
        Its contribution to the diagonal is exactly ``1 / total_area`` at every
        vertex and every timescale — the equilibrium the heat kernel relaxes
        to — so keeping it adds a constant that says only how large the mesh
        is.  It dominates the large timescales, and the constant differs from
        mesh to mesh, so keeping it writes total area into every vertex of
        every mesh being compared.  Chunking is the case where that is
        unavoidable rather than the only case it matters in.

        It has nothing to do with *vertex* area.  The constant eigenvector is
        M-orthonormal, so its square is ``1 / total_area`` at every vertex
        alike, measured constant to 5e-15 on a mesh whose vertex areas span a
        factor of 24.  The claim it once carried here — that the first
        eigenvector is proportional to vertex areas — is true of the
        *symmetrized* operator :math:`M^{-1/2} L M^{-1/2}`, whose first
        eigenvector is :math:`\sqrt{\mathrm{area}}` and whose square is
        therefore proportional to vertex area exactly.  That is a different
        normalization from the one solved here.  See
        [compute_diffused_curvature][meshmash.curvature.compute_diffused_curvature],
        which drops the same mode from its diagonal and adds it back to its
        signal channels, where deleting it would delete the field mean.
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
    seed :
        Seed for the ARPACK starting vector, forwarded to
        [spectral_geometry_filter][meshmash.decompose.spectral_geometry_filter].
        ``None`` lets ARPACK draw its own, so two runs of the same call agree
        only to ``decomposition_dtype``.  An integer makes the result
        reproducible.

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
        seed=seed,
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
