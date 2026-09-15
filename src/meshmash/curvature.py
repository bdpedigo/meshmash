"""Per-vertex curvature and normal-dispersion descriptors, raw and diffused.

Two families live here, built from the same ingredients and diffused by the
same heat kernel.  Curvature says how much the surface bends and in what way;
the normal structure tensor says how much the surface normal *disperses* over
a neighbourhood, which separates a tube from a sheet from a blob.  A corrugated
sheet and a smooth tube can agree on smoothed mean and Gaussian curvature and
disagree here.

Both families are emitted as measures rather than as pointwise values: an
integrated quantity per vertex, in the finite-element sense, with the vertex
area folded in.  That is what makes them diffusible.  Dividing a measure by the
vertex area recovers the pointwise field, and
[compute_diffused_curvature][meshmash.curvature.compute_diffused_curvature] does
exactly that on the way into the filter.
"""

from typing import Callable, NamedTuple, Optional, Union

import numpy as np
from point_cloud_utils import estimate_mesh_vertex_normals
from scipy.sparse import dia_array, sparray

from .decompose import concatenate_filters, get_heat_filter, spectral_geometry_filter
from .laplacian import cotangent_laplacian
from .types import ArrayLike, Mesh, interpret_mesh
from .utils import boundary_vertices

#: The invariants [curvature_invariants][meshmash.curvature.curvature_invariants]
#: emits, in the order it emits them.
CURVATURE_INVARIANT_NAMES = (
    "mean",
    "gauss",
    "k1",
    "k2",
    "shape_index",
    "curvedness",
)

#: The invariants
#: [normal_tensor_invariants][meshmash.curvature.normal_tensor_invariants] emits,
#: in the order it emits them.
NORMAL_TENSOR_INVARIANT_NAMES = ("sheet", "tube", "blob", "trace")

#: The upper triangle of a symmetric 3x3, in the order the normal tensor's six
#: channels are written and read back.
_TENSOR_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def vertex_normals(mesh: Mesh) -> np.ndarray:
    """Unit vertex normals, area-weighted from the face normals.

    A thin wrapper around ``point_cloud_utils.estimate_mesh_vertex_normals``
    with ``weighting_type="area"``, which is also what
    [clean][meshmash.clean] uses for its face normals.

    The sign is whatever the mesh's face winding says, which for a mesh nobody
    has oriented is arbitrary per connected component.  Callers that need a
    consistent sign have to fix it themselves.  See
    [mean_curvature_measure][meshmash.curvature.mean_curvature_measure], which
    fixes it against the mean-curvature direction.

    Parameters
    ----------
    mesh :
        Input mesh.

    Returns
    -------
    :
        Unit normals of shape ``(V, 3)``.  A vertex touching no face, or whose
        face normals cancel exactly, comes back as the zero vector.
    """
    vertices, faces = interpret_mesh(mesh)
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)  # pcu wants int32
    # Area weighting is inherited from the old hand-rolled version, not chosen.
    # Worst-case angle from the exact normal of a sphere: 0.51 deg for "area",
    # 0.05 for "angle".  Worth revisiting, but it moves every downstream feature.
    normals = np.asarray(
        estimate_mesh_vertex_normals(vertices, faces, weighting_type="area"),
        dtype=np.float64,
    )
    # pcu returns a non-finite row where this function promises zero.
    return np.where(np.isfinite(normals).all(axis=1, keepdims=True), normals, 0.0)


def _resolve_laplacian(
    mesh: Mesh,
    laplacian: Optional[tuple[sparray, sparray]],
    robust: bool,
    mollify_factor: float,
) -> tuple[sparray, dia_array]:
    """Reuse a caller's operator, or build one from the mesh."""
    if laplacian is not None:
        return laplacian
    vertices, faces = interpret_mesh(mesh)
    vertices = np.asarray(vertices, dtype=np.float64)
    # The centring below buys nothing and is here only so that every site in
    # this module agrees.  ``L`` and ``M`` are functions of coordinate
    # differences, so a common offset cancels in the first subtraction: 1e-15
    # relative on a float64 mesh, and exactly zero when the coordinates came
    # from float32, whose spare mantissa bits make the shift exact.  See the
    # Notes on mean_curvature_measure for the one site that gains anything.
    return cotangent_laplacian(
        (vertices - vertices.mean(axis=0), np.asarray(faces)),
        robust=robust,
        mollify_factor=mollify_factor,
    )


def _mask(measures: np.ndarray, mesh: Mesh, mask_boundary: bool) -> np.ndarray:
    """Zero the boundary rows, then the non-finite ones.

    Both families do this, and both do it to the measure rather than to the
    field, so that the zero survives division by a vertex area of whatever
    size.  A degenerate face can put a non-finite value anywhere, not only at a
    boundary, and zero is the only value that diffuses harmlessly.
    """
    if mask_boundary:
        measures[boundary_vertices(mesh)] = 0.0
    return np.where(np.isfinite(measures), measures, 0.0)


def mean_curvature_measure(
    mesh: Mesh,
    laplacian: Optional[tuple[sparray, sparray]] = None,
    normals: Optional[np.ndarray] = None,
    mask_boundary: bool = True,
    robust: bool = True,
    mollify_factor: float = 1e-5,
) -> np.ndarray:
    """The integrated mean curvature at each vertex.

    The cotangent Laplacian applied to the coordinates is the mean-curvature
    normal in integrated form, :math:`L V = 2 H \\mathbf{n}` against the vertex
    area.  Projecting it onto the unit normal and halving gives the signed
    measure, so dividing by the vertex area gives mean curvature itself.

    The normals are flipped as a group when the median of their agreement with
    the mean-curvature direction comes out negative.  The winding of a mesh
    nobody has oriented is arbitrary, but the mean-curvature direction is not,
    so the geometry fixes the sign rather than the file.  This is a single
    global vote, not a per-vertex one: a per-vertex flip would make every
    surface read convex.

    Parameters
    ----------
    mesh :
        Input mesh.
    laplacian :
        Pre-built ``(L, M)`` from
        [cotangent_laplacian][meshmash.laplacian.cotangent_laplacian], to share
        one operator across several descriptors.  Built from the mesh when
        ``None``.  Any vertex centring will do: ``L`` is built from coordinate
        differences, so an operator from offset vertices and one from centred
        vertices agree to the last bit or nearly so.
    normals :
        Pre-computed unit vertex normals, shape ``(V, 3)``.  Computed from the
        mesh when ``None``.  The sign vote is applied either way.
    mask_boundary :
        If ``True``, zero the measure at vertices on an open boundary, where
        the vertex area is a partial area and the curvature it implies is
        meaningless.
    robust :
        Passed to [cotangent_laplacian][meshmash.laplacian.cotangent_laplacian]
        when building the operator.  Ignored when ``laplacian`` is given.
    mollify_factor :
        Passed to [cotangent_laplacian][meshmash.laplacian.cotangent_laplacian]
        when building the operator.  Ignored when ``laplacian`` is given.

    Returns
    -------
    :
        Mean-curvature measure of shape ``(V,)``.
    """
    vertices, faces = interpret_mesh(mesh)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    # Centring cancels in exact arithmetic, but the row sums of L land near
    # 1e-15 rather than zero, so an uncentred L @ V carries that residual times
    # the coordinate offset: ~1e-11 relative at a CAVE coordinate of 1e6 nm,
    # against 7e-4 discretisation error on an analytic sphere.  Cheap
    # insurance, not a correctness fix.
    centered = vertices - vertices.mean(axis=0)

    L, _ = _resolve_laplacian(mesh, laplacian, robust, mollify_factor)
    if normals is None:
        normals = vertex_normals((centered, faces))
    normals = np.asarray(normals, dtype=np.float64)

    mean_vector = L.astype(np.float64) @ centered
    normals = orient_normals_by_curvature(normals, mean_vector)

    measure = 0.5 * (mean_vector * normals).sum(axis=1)
    return _mask(measure, (vertices, faces), mask_boundary)


def orient_normals_by_curvature(
    normals: np.ndarray, mean_curvature_vector: np.ndarray
) -> np.ndarray:
    """Flip a whole normal field into agreement with the mean-curvature normal.

    Parameters
    ----------
    normals :
        Vertex normals, shape ``(V, 3)``.  Need not be unit length.
    mean_curvature_vector :
        The mean-curvature normal ``L @ V``, shape ``(V, 3)``.

    Returns
    -------
    :
        Either ``normals`` or ``-normals``, whichever agrees with
        ``mean_curvature_vector`` at more than half of the vertices where both
        are non-zero.  Returns ``normals`` unchanged when no vertex qualifies.
    """
    vector_norm = np.linalg.norm(mean_curvature_vector, axis=1)
    normal_norm = np.linalg.norm(normals, axis=1)
    usable = (vector_norm > 0) & (normal_norm > 0)
    if not usable.any():
        return normals
    cosines = (
        mean_curvature_vector[usable]
        / vector_norm[usable, None]
        * (normals[usable] / normal_norm[usable, None])
    ).sum(axis=1)
    if np.median(cosines) < 0:
        return -normals
    return normals


def gaussian_curvature_measure(mesh: Mesh, mask_boundary: bool = True) -> np.ndarray:
    """The integrated Gaussian curvature at each vertex, as the angle defect.

    The angle defect is :math:`2\\pi` minus the interior angles meeting at a
    vertex.  By the discrete Gauss-Bonnet theorem this is exactly the integral
    of Gaussian curvature over that vertex's cell, with no discretisation
    choice left open, so dividing by the vertex area gives Gaussian curvature.

    Parameters
    ----------
    mesh :
        Input mesh.
    mask_boundary :
        If ``True``, zero the measure at vertices on an open boundary.  There
        the angles do not close, so the defect reads as spurious curvature.
        ``gpytoolbox.angle_defect`` already returns zero at a boundary vertex,
        so this only matters for keeping the two families masked alike.

    Returns
    -------
    :
        Gaussian-curvature measure of shape ``(V,)``.
    """
    from gpytoolbox import angle_defect

    vertices, faces = interpret_mesh(mesh)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    # A no-op: the angle defect reads only angles, which no offset can move.
    # Kept to match mean_curvature_measure, whose Notes give the reasoning.
    measure = np.asarray(
        angle_defect(vertices - vertices.mean(axis=0), faces), dtype=np.float64
    )
    return _mask(measure, (vertices, faces), mask_boundary)


def normal_tensor_measure(
    mesh: Mesh,
    normals: Optional[np.ndarray] = None,
    areas: Optional[np.ndarray] = None,
    mask_boundary: bool = True,
    robust: bool = True,
    mollify_factor: float = 1e-5,
) -> np.ndarray:
    """The integrated normal structure tensor at each vertex.

    The outer product :math:`\\mathbf{n} \\mathbf{n}^T` of the unit normal,
    area-weighted, with its six unique entries in the order
    ``(xx, yy, zz, xy, xz, yz)``.  Undiffused this is rank one at every vertex
    and says nothing.  Diffused it becomes the second moment of the normal
    direction over a neighbourhood, and its eigenvalue spread separates a tube
    from a sheet from a blob.  See
    [normal_tensor_invariants][meshmash.curvature.normal_tensor_invariants].

    Squaring before averaging is the whole point, and the order does not
    commute: the average of the normals themselves is a single vector, which
    is rank one whatever the surface does, whereas the average of the outer
    products keeps the spread.

    Parameters
    ----------
    mesh :
        Input mesh.
    normals :
        Pre-computed vertex normals, shape ``(V, 3)``.  Computed from the mesh
        when ``None``.  Normalised here either way.
    areas :
        Pre-computed vertex areas, shape ``(V,)``, as the diagonal of the mass
        matrix.  Computed from the mesh when ``None``.
    mask_boundary :
        If ``True``, zero the measure at vertices on an open boundary.
    robust :
        Passed to [cotangent_laplacian][meshmash.laplacian.cotangent_laplacian]
        when computing areas.  Ignored when ``areas`` is given.
    mollify_factor :
        Passed to [cotangent_laplacian][meshmash.laplacian.cotangent_laplacian]
        when computing areas.  Ignored when ``areas`` is given.

    Returns
    -------
    :
        Normal-tensor measure of shape ``(V, 6)``.
    """
    vertices, faces = interpret_mesh(mesh)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    # A no-op: the normals and the areas below read coordinate differences only.
    # Kept to match mean_curvature_measure, whose Notes give the reasoning.
    centered = vertices - vertices.mean(axis=0)

    if normals is None:
        normals = vertex_normals((centered, faces))
    normals = np.asarray(normals, dtype=np.float64)
    if areas is None:
        _, M = cotangent_laplacian(
            (centered, faces), robust=robust, mollify_factor=mollify_factor
        )
        areas = np.asarray(M.diagonal(), dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)

    lengths = np.linalg.norm(normals, axis=1)
    unit = np.where(
        lengths[:, None] > 0,
        normals / np.where(lengths > 0, lengths, 1.0)[:, None],
        0.0,
    )
    tensor = np.column_stack([unit[:, i] * unit[:, j] for i, j in _TENSOR_PAIRS])
    return _mask(areas[:, None] * tensor, (centered, faces), mask_boundary)


def curvature_invariants(mean: ArrayLike, gauss: ArrayLike) -> np.ndarray:
    """Six local shape descriptors from mean and Gaussian curvature.

    Pointwise and unit-agnostic: the inputs are curvatures, not measures, and
    the outputs carry whatever length unit they came in with.  ``mean``,
    ``k1``, ``k2`` and ``curvedness`` are in inverse length, ``gauss`` in
    inverse length squared, and ``shape_index`` is dimensionless in
    ``[-1, 1]``.

    The principal curvatures come from inverting the symmetric functions,
    :math:`\\kappa_{1,2} = H \\pm \\sqrt{H^2 - K}`.  The shape index and the
    curvedness are the polar coordinates of :math:`(\\kappa_1, \\kappa_2)`
    from [1]: the shape index is the angle, and says *what* shape without
    regard to scale, while the curvedness is the radius, and says how sharply
    curved without regard to kind.

    Parameters
    ----------
    mean :
        Mean curvature :math:`H = (\\kappa_1 + \\kappa_2) / 2`, shape ``(V,)``.
    gauss :
        Gaussian curvature :math:`K = \\kappa_1 \\kappa_2`, shape ``(V,)``.

    Returns
    -------
    :
        Array of shape ``(V, 6)``, with columns named by
        [CURVATURE_INVARIANT_NAMES][meshmash.curvature.CURVATURE_INVARIANT_NAMES].

    Notes
    -----
    :math:`H^2 - K` is non-negative for any real surface, being
    :math:`((\\kappa_1 - \\kappa_2) / 2)^2`, but discrete mean and Gaussian
    curvature come from different estimators and can disagree by enough to
    make it negative.  It is clipped at zero, which reports an umbilic point
    where the two estimators fell out of step.

    The shape index is written with a two-argument arctangent so that an
    umbilic point, where the denominator is zero, gives the limit rather than
    a division by zero.  For a convex umbilic it returns ``+1``.

    References
    ----------
    [1] J. J. Koenderink and A. J. van Doorn, "Surface shape and curvature
    scales", Image and Vision Computing, 10(8):557-564, 1992.
    """
    mean = np.asarray(mean, dtype=np.float64)
    gauss = np.asarray(gauss, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        deviation = np.sqrt(np.clip(mean**2 - gauss, 0.0, None))
        return np.column_stack(
            [
                mean,
                gauss,
                mean + deviation,
                mean - deviation,
                (2.0 / np.pi) * np.arctan2(mean, deviation),
                np.sqrt(np.clip(2.0 * mean**2 - gauss, 0.0, None)),
            ]
        )


def normal_tensor_invariants(tensor: ArrayLike) -> np.ndarray:
    """Shape fractions and trace of a diffused normal structure tensor.

    Takes the six unique entries of a symmetric 3x3 per vertex, in the order
    [normal_tensor_measure][meshmash.curvature.normal_tensor_measure] writes
    them, and returns the eigenvalue shape fractions of [1] plus the trace.

    With eigenvalues :math:`\\lambda_1 \\ge \\lambda_2 \\ge \\lambda_3 \\ge 0`
    summing to :math:`S`, the fractions are
    :math:`(\\lambda_1 - \\lambda_2) / S`,
    :math:`2 (\\lambda_2 - \\lambda_3) / S` and
    :math:`3 \\lambda_3 / S`.  They are non-negative and sum to one.  One
    dominant normal direction is a locally flat neighbourhood, so it reads
    ``sheet``; two, a neighbourhood curving one way, reads ``tube``; three,
    curving every way, reads ``blob``.

    The ``trace`` column is a numerics canary rather than a shape descriptor.
    The undiffused tensor has unit trace at every vertex, diffusion conserves
    it, so anything other than one means mass went missing.  A masked boundary
    is the expected place for that.

    Parameters
    ----------
    tensor :
        Tensor entries of shape ``(V, 6)``, ordered
        ``(xx, yy, zz, xy, xz, yz)``.

    Returns
    -------
    :
        Array of shape ``(V, 4)``, with columns named by
        [NORMAL_TENSOR_INVARIANT_NAMES][meshmash.curvature.NORMAL_TENSOR_INVARIANT_NAMES].

    Notes
    -----
    The eigenvalues are clipped at zero.  An average of outer products is
    positive semidefinite in exact arithmetic, and the smallest eigenvalue of a
    nearly rank-deficient one comes back slightly negative from rounding.

    Rows that are not finite come back as ``NaN`` rather than raising, because
    [eigvalsh][numpy.linalg.eigvalsh] refuses non-finite input and a caller
    stitching chunks together has rows it never computed.

    References
    ----------
    [1] C.-F. Westin et al., "Processing and visualization for diffusion tensor
    MRI", Medical Image Analysis, 6(2):93-108, 2002.
    """
    tensor = np.asarray(tensor, dtype=np.float64)
    trace = tensor[:, 0] + tensor[:, 1] + tensor[:, 2]

    out = np.full((len(tensor), 4), np.nan)
    out[:, 3] = trace
    finite = np.isfinite(tensor).all(axis=1)
    if not finite.any():
        return out

    rows = tensor[finite]
    matrices = np.empty((len(rows), 3, 3), dtype=np.float64)
    for column, (i, j) in enumerate(_TENSOR_PAIRS):
        matrices[:, i, j] = rows[:, column]
        matrices[:, j, i] = rows[:, column]

    # Ascending from eigvalsh; the fractions want descending.
    lam = np.clip(np.linalg.eigvalsh(matrices)[:, ::-1], 0.0, None)
    total = lam.sum(axis=1)
    positive = total > 0
    safe = np.where(positive, total, 1.0)
    out[finite, 0] = np.where(positive, (lam[:, 0] - lam[:, 1]) / safe, 0.0)
    out[finite, 1] = np.where(positive, 2.0 * (lam[:, 1] - lam[:, 2]) / safe, 0.0)
    out[finite, 2] = np.where(positive, 3.0 * lam[:, 2] / safe, 0.0)
    return out


class DiffusedCurvatureResult(NamedTuple):
    """What [compute_diffused_curvature][meshmash.curvature.compute_diffused_curvature] returns."""

    #: Filtered kernel diagonal, shape ``(V, D)``.  The heat kernel signature
    #: at ``scales`` when no ``diagonal_filter`` was given.
    diagonal: np.ndarray
    #: Undiffused curvature invariants, shape ``(V, 6)``.
    raw_curvature: np.ndarray
    #: Diffused curvature invariants, shape ``(V, n_scales, 6)``.
    curvature: np.ndarray
    #: Diffused normal-tensor invariants, shape ``(V, n_scales, 4)``.
    normal_tensor: np.ndarray


def compute_diffused_curvature(
    mesh: Mesh,
    scales: ArrayLike,
    diagonal_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    max_eigenvalue: float = 1e-8,
    band_size: int = 50,
    truncate_extra: bool = True,
    drop_first: bool = True,
    robust: bool = True,
    mollify_factor: float = 1e-5,
    decomposition_dtype: Optional[np.dtype] = np.float64,
    signal_dtype: np.dtype = np.float64,
    verbose: Union[bool, int] = False,
) -> DiffusedCurvatureResult:
    """Curvature and normal-tensor descriptors at several scales, off one solve.

    Smoothing a curvature estimate is the point, not a cleanup step: a mesh has
    no intrinsic scale, and what a surface *is* depends on how far away one
    stands from it.  A dendritic spine is a bump at one scale and part of a
    shaft at another.  This diffuses both families through the true heat
    kernel, :math:`\\Phi e^{-t \\Lambda} \\Phi^T`, at each of ``scales``, which
    smooths over roughly ``sqrt(t)`` of surface.

    All of it rides one eigendecomposition.  The kernel diagonal, the six
    curvature channels and the six tensor channels are accumulated band by band
    from the same eigenpairs, so adding a family costs another matrix product
    and not another solve.  Pass ``diagonal_filter`` to read the diagonal at
    different timescales from the ones the signals are diffused at, which is how
    a heat kernel signature comes back from the same call.

    Parameters
    ----------
    mesh :
        Input mesh.
    scales :
        Diffusion timescales for the signal channels, shape ``(T,)``.
    diagonal_filter :
        Filter for the kernel diagonal, as built by
        [get_hks_filter][meshmash.decompose.get_hks_filter].  When ``None``, the
        diagonal comes back at ``scales``, which is the heat kernel signature
        there.
    max_eigenvalue :
        Eigenvalue to decompose up to.  Truncating the spectrum is a spatial
        cutoff: modes above it vary faster than the smallest timescale can see.
    band_size :
        Eigenvalues to compute at a time.  Does not change the result.
    truncate_extra :
        If ``True``, cut the spectrum at ``max_eigenvalue`` exactly.
    drop_first :
        If ``True``, drop the constant eigenpair from the diagonal, matching
        [compute_hks][meshmash.decompose.compute_hks].  The signal channels are
        compensated for it, so this does not change them.  See Notes.
    robust :
        Passed to [cotangent_laplacian][meshmash.laplacian.cotangent_laplacian].
    mollify_factor :
        Passed to [cotangent_laplacian][meshmash.laplacian.cotangent_laplacian].
    decomposition_dtype :
        Dtype for the decomposition and the diagonal accumulation.
    signal_dtype :
        Dtype for the signal accumulation.
    verbose :
        If >0, print progress.  Higher values give more.

    Returns
    -------
    :
        A [DiffusedCurvatureResult][meshmash.curvature.DiffusedCurvatureResult].

    Notes
    -----
    **Measures in, fields across the boundary.**  Both families are built as
    measures, and are divided by the vertex areas before they are handed to
    [spectral_geometry_filter][meshmash.decompose.spectral_geometry_filter],
    which weights its signals by the mass matrix itself.  With the lumped
    diagonal mass matrix that both
    [cotangent_laplacian][meshmash.laplacian.cotangent_laplacian] paths return,
    that reconstructs the measures exactly, so each projection is
    :math:`\\phi^T b` and the diffused field is
    :math:`\\Phi e^{-t \\Lambda} \\Phi^T b`.

    **The dropped constant mode is added back.**  Deleting the constant
    eigenvector deletes the signal's area-weighted mean, which is wrong for a
    diffused field rather than merely conventional: the heat kernel conserves
    mass, and without the constant mode the normal tensor's trace sags below
    one everywhere instead of only at masked vertices.  Its contribution is the
    same at every timescale, since :math:`e^{-t \\cdot 0} = 1`, so it is added
    back as the field's mean after filtering.

    **The heat bank's diagonal is computed and discarded.**
    [spectral_geometry_filter][meshmash.decompose.spectral_geometry_filter]
    applies one filter bank to both of its outputs, and this function hands it
    a stacked bank whose second half exists only to diffuse the signals.  The
    diagonal of that half is computed and dropped, at the cost of one extra
    matrix product per band, which is small beside the solve.
    """
    vertices, faces = interpret_mesh(mesh)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    # Reaches a result only through the ``L @ V`` inside mean_curvature_measure,
    # and there by about 1e-11 relative.  See the Notes on that function.
    centered = vertices - vertices.mean(axis=0)

    L, M = cotangent_laplacian(
        (centered, faces), robust=robust, mollify_factor=mollify_factor
    )
    areas = np.asarray(M.diagonal(), dtype=np.float64)
    safe_areas = np.where(areas > 0, areas, 1.0)

    # One normal field and one boundary mask, shared by both families.
    normals = vertex_normals((centered, faces))
    normals = orient_normals_by_curvature(normals, L.astype(np.float64) @ centered)
    measures = np.column_stack(
        [
            mean_curvature_measure(
                (centered, faces), laplacian=(L, M), normals=normals
            ),
            gaussian_curvature_measure((centered, faces)),
            normal_tensor_measure((centered, faces), normals=normals, areas=areas),
        ]
    )
    fields = measures / safe_areas[:, None]

    scales = np.asarray(scales, dtype=np.float64)
    heat_filter = get_heat_filter(scales, dtype=decomposition_dtype or np.float64)
    n_diagonal = (
        len(scales)
        if diagonal_filter is None
        else np.asarray(diagonal_filter(np.array([1.0]))).shape[0]
    )

    diagonal, filtered = spectral_geometry_filter(
        (L, M),
        concatenate_filters(diagonal_filter, heat_filter),
        max_eigenvalue=max_eigenvalue,
        band_size=band_size,
        truncate_extra=truncate_extra,
        drop_first=drop_first,
        decomposition_dtype=decomposition_dtype,
        signals=fields,
        signal_dtype=signal_dtype,
        verbose=verbose,
    )
    diffused = filtered[:, -len(scales) :, :]
    if drop_first:
        diffused = diffused + measures.sum(axis=0) / areas.sum()

    raw = curvature_invariants(
        np.where(areas > 0, measures[:, 0] / safe_areas, 0.0),
        np.where(areas > 0, measures[:, 1] / safe_areas, 0.0),
    )
    curvature = np.stack(
        [
            curvature_invariants(diffused[:, index, 0], diffused[:, index, 1])
            for index in range(len(scales))
        ],
        axis=1,
    )
    normal_tensor = np.stack(
        [
            normal_tensor_invariants(diffused[:, index, 2:])
            for index in range(len(scales))
        ],
        axis=1,
    )
    return DiffusedCurvatureResult(
        diagonal=diagonal[:, :n_diagonal],
        raw_curvature=raw,
        curvature=curvature,
        normal_tensor=normal_tensor,
    )
