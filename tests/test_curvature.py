"""Curvature and normal dispersion, checked against shapes with known answers.

A descriptor tested only on a real mesh is tested against no answer. A sphere
and a cylinder both have closed forms for everything here, including for the
diffused versions, because the coordinate and normal fields on them are
eigenfunctions of their own Laplacians. Those closed forms are what these
assert against.
"""

import numpy as np
import pytest
import pyvista as pv
from scipy.linalg import eigh

from meshmash import (
    CURVATURE_INVARIANT_NAMES,
    NORMAL_TENSOR_INVARIANT_NAMES,
    boundary_vertices,
    compute_diffused_curvature,
    curvature_invariants,
    gaussian_curvature_measure,
    mean_curvature_measure,
    normal_tensor_invariants,
    normal_tensor_measure,
    vertex_normals,
)
from meshmash.decompose import get_hks_filter
from meshmash.laplacian import cotangent_laplacian
from meshmash.utils import poly_to_mesh

RADIUS = 1000.0
MAX_EIGENVALUE = 1e-4

#: A dendrite-scale tube: long enough that its middle is many diffusion
#: lengths from either end at the largest timescale used here.
TUBE_RADIUS = 400.0
TUBE_LENGTH = 12_000.0


@pytest.fixture(scope="module")
def sphere():
    return poly_to_mesh(
        pv.Sphere(radius=RADIUS, theta_resolution=40, phi_resolution=40).triangulate()
    )


@pytest.fixture(scope="module")
def small_sphere():
    """Small enough for a dense reference decomposition, and jittered.

    A round sphere's eigenvalues have multiplicity 2l+1, and two solves return
    different bases inside a degenerate eigenspace. A constant field such as a
    sphere's mean curvature does not notice, but the normal tensor does, so
    every test that compares two decompositions needs a simple spectrum.
    """
    poly = pv.Sphere(radius=RADIUS, theta_resolution=16, phi_resolution=16)
    vertices, faces = poly_to_mesh(poly.triangulate())
    rng = np.random.default_rng(3)
    scaling = 1.0 + 0.05 * rng.normal(size=(len(vertices), 1))
    return (np.asarray(vertices) * scaling, np.asarray(faces))


@pytest.fixture(scope="module")
def tube():
    """An open-ended tube of known radius, with rings evenly spaced along it.

    ``clean`` is not cosmetic: the tube filter leaves the seam ring duplicated,
    and a pair of coincident vertices each carrying a full vertex area doubles
    the mass there, which shows up as a normal-tensor trace of 2.
    """
    poly = pv.Line(
        (0, 0, -TUBE_LENGTH / 2), (0, 0, TUBE_LENGTH / 2), resolution=120
    ).tube(radius=TUBE_RADIUS, n_sides=40, capping=False)
    return poly_to_mesh(poly.triangulate().clean(tolerance=1e-6).triangulate())


@pytest.fixture(scope="module")
def hemisphere():
    """Half a sphere, so its rim is a cut through curvature rather than an edge."""
    poly = pv.Sphere(
        radius=RADIUS,
        theta_resolution=40,
        phi_resolution=40,
        start_phi=0,
        end_phi=90,
    )
    return poly_to_mesh(poly.triangulate().clean(tolerance=1e-6).triangulate())


@pytest.fixture(scope="module")
def plane():
    """A flat grid, so every edge vertex is a boundary vertex."""
    poly = pv.Plane(i_size=4000, j_size=4000, i_resolution=20, j_resolution=20)
    return poly_to_mesh(poly.triangulate())


def vertex_areas(mesh):
    _, M = cotangent_laplacian(mesh, robust=True)
    return np.asarray(M.diagonal())


def interior(mesh):
    mask = np.ones(len(mesh[0]), dtype=bool)
    mask[boundary_vertices(mesh)] = False
    return mask


# --- the primitives -------------------------------------------------------


def test_vertex_normals_are_unit_and_radial_on_a_sphere(sphere):
    normals = vertex_normals(sphere)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=1e-10)
    cosines = (normals * (np.asarray(sphere[0]) / RADIUS)).sum(axis=1)
    # Sign is the winding's, magnitude is the geometry's.
    assert np.abs(cosines).min() > 0.99


def test_a_sphere_reads_its_own_radius(sphere):
    """Everything a sphere's curvature says is fixed by its radius."""
    areas = vertex_areas(sphere)
    invariants = curvature_invariants(
        mean_curvature_measure(sphere) / areas,
        gaussian_curvature_measure(sphere) / areas,
    )
    columns = dict(zip(CURVATURE_INVARIANT_NAMES, invariants.T))

    assert np.median(columns["mean"]) * RADIUS == pytest.approx(1.0, abs=0.01)
    assert np.median(columns["gauss"]) * RADIUS**2 == pytest.approx(1.0, abs=0.01)
    assert np.median(columns["curvedness"]) * RADIUS == pytest.approx(1.0, abs=0.01)

    # Both principal curvatures are 1/R, so every point is umbilic and the
    # shape index sits at the end of its range. Not to machine precision: mean
    # and Gaussian curvature come from two different discrete estimators, the
    # cotangent Laplacian and the angle defect, and they disagree by a few
    # percent on the stretched triangles near this sphere's poles. Umbilic is
    # the shape that disagreement shows up in, since it is the shape where the
    # two principal curvatures have no real gap to resolve.
    spread = np.abs(columns["k1"] - columns["k2"]) / (columns["k1"] + columns["k2"])
    assert np.median(spread) < 0.02
    assert np.median(columns["shape_index"]) > 0.95


def test_the_mean_curvature_sign_follows_the_geometry_not_the_winding(sphere):
    """A sphere reads convex either way its faces happen to be wound."""
    flipped = (np.asarray(sphere[0]), np.asarray(sphere[1])[:, ::-1])
    np.testing.assert_allclose(
        mean_curvature_measure(flipped), mean_curvature_measure(sphere), rtol=1e-6
    )
    assert np.median(mean_curvature_measure(sphere)) > 0


def test_a_plane_is_flat_inside_and_masked_at_the_rim(plane):
    areas = vertex_areas(plane)
    inside = interior(plane)
    rim = ~inside

    # A 21x21 grid has 80 vertices around its edge.
    assert rim.sum() == 80

    mean = mean_curvature_measure(plane) / areas
    gauss = gaussian_curvature_measure(plane) / areas
    assert np.abs(mean[inside]).max() < 1e-9
    assert np.abs(gauss[inside]).max() < 1e-9
    np.testing.assert_array_equal(mean[rim], 0.0)
    np.testing.assert_array_equal(gauss[rim], 0.0)


def test_an_unmasked_boundary_reports_the_wrong_curvature(hemisphere):
    """What the mask is for, on a rim whose true curvature is known.

    Every vertex of this hemisphere sits on a sphere of radius R, rim included,
    so the answer everywhere is 1/R. A rim vertex sees half of its
    neighbourhood, and reads about half the mean curvature and none of the
    Gaussian curvature. The error is not noise that diffusion averages away,
    it is a bias along the whole cut, so it is zeroed instead.

    ``gpytoolbox.angle_defect`` already returns zero at a boundary vertex, so
    the Gaussian measure is masked there whether this asks for it or not.
    """
    areas = vertex_areas(hemisphere)
    rim = ~interior(hemisphere)

    mean = mean_curvature_measure(hemisphere, mask_boundary=False) / areas
    gauss = gaussian_curvature_measure(hemisphere, mask_boundary=False) / areas
    assert np.median(mean[~rim]) * RADIUS == pytest.approx(1.0, abs=0.01)
    assert np.median(mean[rim]) * RADIUS == pytest.approx(0.5, abs=0.05)
    assert np.median(gauss[~rim]) * RADIUS**2 == pytest.approx(1.0, abs=0.01)
    np.testing.assert_array_equal(gauss[rim], 0.0)

    np.testing.assert_array_equal(mean_curvature_measure(hemisphere)[rim], 0.0)


def test_the_undiffused_normal_tensor_is_rank_one(sphere):
    """Which is why this family has no raw columns worth emitting."""
    areas = vertex_areas(sphere)
    invariants = normal_tensor_invariants(
        normal_tensor_measure(sphere) / areas[:, None]
    )
    columns = dict(zip(NORMAL_TENSOR_INVARIANT_NAMES, invariants.T))
    np.testing.assert_allclose(columns["sheet"], 1.0, atol=1e-8)
    np.testing.assert_allclose(columns["tube"], 0.0, atol=1e-8)
    np.testing.assert_allclose(columns["trace"], 1.0, rtol=1e-10)


def test_the_normal_tensor_ignores_the_winding(tube):
    """The outer product erases the normal's sign, so no orientation vote is needed.

    To rounding rather than to the bit: reversing the winding negates each
    face's cross product, and the normals are summed in the same order but with
    the opposite sign, which is not the same floating-point sum.
    """
    flipped = (np.asarray(tube[0]), np.asarray(tube[1])[:, ::-1])
    np.testing.assert_allclose(
        normal_tensor_measure(flipped), normal_tensor_measure(tube), rtol=1e-12
    )


def test_curvature_invariants_on_known_curvatures():
    #                 sphere    cylinder   plane    saddle
    k1 = np.array([2.0, 3.0, 0.0, 1.0])
    k2 = np.array([2.0, 0.0, 0.0, -1.0])
    invariants = curvature_invariants((k1 + k2) / 2, k1 * k2)
    columns = dict(zip(CURVATURE_INVARIANT_NAMES, invariants.T))

    np.testing.assert_allclose(columns["k1"], k1)
    np.testing.assert_allclose(columns["k2"], k2)
    np.testing.assert_allclose(columns["curvedness"], np.sqrt((k1**2 + k2**2) / 2))
    # Koenderink's landmark values: a convex umbilic, a ridge, and a symmetric
    # saddle, which is the self-complementary shape at the centre of the range.
    np.testing.assert_allclose(columns["shape_index"], [1.0, 0.5, 0.0, 0.0], atol=1e-12)


def test_curvature_invariants_clip_an_impossible_pair():
    """H^2 < K cannot happen on a surface, but two estimators can disagree."""
    invariants = curvature_invariants([0.0], [1.0])
    np.testing.assert_allclose(invariants[0, 2], invariants[0, 3])


def test_normal_tensor_invariants_on_known_tensors():
    # xx, yy, zz, xy, xz, yz for one, two and three equal normal directions.
    tensors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            [1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0],
        ]
    )
    columns = dict(
        zip(NORMAL_TENSOR_INVARIANT_NAMES, normal_tensor_invariants(tensors).T)
    )
    np.testing.assert_allclose(columns["sheet"], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(columns["tube"], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(columns["blob"], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(columns["trace"], 1.0)


def test_normal_tensor_invariants_pass_non_finite_rows_through():
    """A caller stitching chunks together has rows it never computed."""
    tensors = np.array([[1.0, 0, 0, 0, 0, 0], [np.nan] * 6])
    invariants = normal_tensor_invariants(tensors)
    assert np.isfinite(invariants[0]).all()
    assert np.isnan(invariants[1]).all()


# --- the fused featurizer -------------------------------------------------


def test_the_diffusion_is_the_truncated_spectral_diffusion(small_sphere):
    """The whole construction against a dense decomposition of the same operator.

    The timescales are deep enough that the truncated spectrum has converged,
    meaning ``exp(-t * max_eigenvalue)`` is below machine precision. They have
    to be. The band-by-band solve cuts the spectrum a band at a time and keeps
    the first eigenvalue past ``max_eigenvalue``, where a dense solve cuts at
    it exactly, so the two bases differ by one mode at the edge. At a shallow
    timescale that mode still carries weight and the comparison measures the
    off-by-one rather than the diffusion.
    """
    scales = np.array([3e5, 1e6])
    result = compute_diffused_curvature(
        small_sphere, scales, max_eigenvalue=MAX_EIGENVALUE, drop_first=False
    )

    L, M = cotangent_laplacian(small_sphere, robust=True)
    areas = np.asarray(M.diagonal())
    measure = mean_curvature_measure(small_sphere, laplacian=(L, M))
    eigenvalues, eigenvectors = eigh(L.toarray(), M.toarray())
    keep = eigenvectors[:, eigenvalues <= MAX_EIGENVALUE]
    kept_eigenvalues = eigenvalues[eigenvalues <= MAX_EIGENVALUE]
    assert np.exp(-scales.min() * MAX_EIGENVALUE) < 1e-12

    for index, scale in enumerate(scales):
        expected = keep @ (np.exp(-scale * kept_eigenvalues) * (keep.T @ measure))
        np.testing.assert_allclose(
            result.curvature[:, index, 0],
            expected,
            rtol=1e-8,
            atol=1e-10 * np.abs(expected).max(),
        )
    # The raw column is the undiffused field, not a diffused one.
    np.testing.assert_allclose(result.raw_curvature[:, 0], measure / areas, rtol=1e-10)


def test_dropping_the_constant_mode_leaves_the_fields_alone(small_sphere):
    """The compensation is what makes `drop_first` a diagonal-only choice."""
    scales = np.array([2e4, 1e5])
    kwargs = dict(max_eigenvalue=MAX_EIGENVALUE)
    kept = compute_diffused_curvature(small_sphere, scales, drop_first=False, **kwargs)
    dropped = compute_diffused_curvature(
        small_sphere, scales, drop_first=True, **kwargs
    )
    np.testing.assert_allclose(dropped.curvature, kept.curvature, rtol=1e-5)
    np.testing.assert_allclose(
        dropped.normal_tensor, kept.normal_tensor, rtol=1e-5, atol=1e-6
    )


def test_the_tensor_trace_is_conserved_on_a_closed_mesh(sphere):
    """Diffusion moves mass around and does not create or destroy it."""
    result = compute_diffused_curvature(
        sphere, np.geomspace(1e4, 2.5e5, 4), max_eigenvalue=MAX_EIGENVALUE
    )
    np.testing.assert_allclose(result.normal_tensor[:, :, 3], 1.0, atol=1e-3)


def test_the_shape_fractions_sum_to_one(sphere):
    result = compute_diffused_curvature(
        sphere, np.geomspace(1e4, 2.5e5, 4), max_eigenvalue=MAX_EIGENVALUE
    )
    np.testing.assert_allclose(
        result.normal_tensor[:, :, :3].sum(axis=-1), 1.0, atol=1e-6
    )


def test_a_sphere_reads_sheet_then_blob(sphere):
    """Scale reads size. A small patch of a sphere is flat, the whole of it is not."""
    scales = np.array([1e4, 2.5e5])
    result = compute_diffused_curvature(sphere, scales, max_eigenvalue=MAX_EIGENVALUE)
    assert np.median(result.normal_tensor[:, 0, 0]) > 0.9
    assert np.median(result.normal_tensor[:, 1, 2]) > 0.7
    assert np.median(result.normal_tensor[:, 1, 0]) < 0.3


def test_a_tube_reads_sheet_then_tube_at_its_own_caliber(tube):
    """Exactly, not approximately, because a cylinder has a closed form.

    The transverse normal components are the cylinder's n=2 angular
    eigenfunctions, eigenvalue 4/R^2, so mid-shaft the diffused tensor has
    eigenvalues (1 +/- exp(-4t/R^2))/2 and zero. The sheet fraction is their
    difference, exp(-4t/R^2), and the tube fraction is the rest. The flip
    between them lands at sqrt(t) = R sqrt(ln 2) / 2, about 0.42 R.
    """
    scales = np.array([112.0, 224.0, 800.0]) ** 2
    axis = np.asarray(tube[0])[:, 2]
    mid_shaft = np.abs(axis) < TUBE_LENGTH / 6
    assert mid_shaft.sum() > 500

    result = compute_diffused_curvature(tube, scales, max_eigenvalue=MAX_EIGENVALUE)
    for index, scale in enumerate(scales):
        expected = np.exp(-4.0 * scale / TUBE_RADIUS**2)
        sheet, tube_fraction, blob = (
            np.median(result.normal_tensor[mid_shaft, index, column])
            for column in range(3)
        )
        assert sheet == pytest.approx(expected, abs=0.005)
        assert tube_fraction == pytest.approx(1.0 - expected, abs=0.005)
        assert blob < 0.005


def test_a_tube_keeps_its_mean_curvature_under_diffusion(tube):
    """Mid-shaft a cylinder's curvature is constant, and diffusing it changes nothing."""
    axis = np.asarray(tube[0])[:, 2]
    mid_shaft = np.abs(axis) < TUBE_LENGTH / 6
    result = compute_diffused_curvature(
        tube, np.array([112.0, 800.0]) ** 2, max_eigenvalue=MAX_EIGENVALUE
    )
    for index in range(2):
        mean = np.median(result.curvature[mid_shaft, index, 0])
        gauss = np.median(result.curvature[mid_shaft, index, 1])
        assert mean * TUBE_RADIUS == pytest.approx(0.5, abs=0.005)
        assert abs(gauss) * TUBE_RADIUS**2 < 0.005


def test_the_diagonal_is_the_heat_kernel_signature(sphere):
    """What `diagonal_filter` buys: one decomposition serving both readings.

    The sphere is jittered because a round one has degenerate eigenspaces,
    where two independent band-by-band runs return different bases and so
    different features. The point under test is the fused bank, not ARPACK.
    """
    from meshmash import compute_hks

    vertices, faces = np.asarray(sphere[0]), np.asarray(sphere[1])
    rng = np.random.default_rng(3)
    jittered = (vertices * (1.0 + 0.05 * rng.normal(size=(len(vertices), 1))), faces)

    kwargs = dict(max_eigenvalue=MAX_EIGENVALUE, truncate_extra=True, drop_first=True)
    result = compute_diffused_curvature(
        jittered,
        np.geomspace(1e4, 2.5e5, 4),
        diagonal_filter=get_hks_filter(2.5e5, 1e4, 6),
        **kwargs,
    )
    expected = compute_hks(jittered, t_min=1e4, t_max=2.5e5, n_components=6, **kwargs)
    assert result.diagonal.shape == expected.shape
    np.testing.assert_allclose(result.diagonal, expected, rtol=1e-8)


def test_without_a_diagonal_filter_the_diagonal_comes_back_at_the_scales(sphere):
    scales = np.geomspace(1e4, 2.5e5, 5)
    result = compute_diffused_curvature(sphere, scales, max_eigenvalue=MAX_EIGENVALUE)
    assert result.diagonal.shape == (len(sphere[0]), len(scales))
    assert result.curvature.shape == (len(sphere[0]), len(scales), 6)
    assert result.normal_tensor.shape == (len(sphere[0]), len(scales), 4)
    assert result.raw_curvature.shape == (len(sphere[0]), 6)
