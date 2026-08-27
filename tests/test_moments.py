"""Heat-kernel moments: the signal filter's algebra, then the invariants.

Two things can be wrong independently. The band-by-band accumulation of
``K_t f`` is linear algebra with a right answer that a dense heat kernel
built from the same eigenpairs gives exactly, so it is checked against that.
The invariants on top are checked for the properties they claim — invariance
under rigid motion — and on geometry whose answer is known by symmetry.
"""

import numpy as np
import pytest
import pyvista as pv

from meshmash import (
    HEAT_KERNEL_MOMENT_NAMES,
    compute_heat_kernel_moments,
    heat_kernel_moment_names,
    hemisphere_profile,
    revolve_profile,
    tube_profile,
    vertex_normals,
)
from meshmash.decompose import get_hks_filter, spectral_geometry_filter
from meshmash.laplacian import cotangent_laplacian
from meshmash.utils import poly_to_mesh

#: A sphere this size in nm puts its eigenvalues, l(l+1)/R^2, in the same
#: range a dendrite chunk's sit in, so the timescales below are the ones the
#: real featurizer uses rather than numbers invented for a unit sphere.
RADIUS = 1000.0
MAX_EIGENVALUE = 1e-4
T_MIN, T_MAX = 1e4, 2.5e5


@pytest.fixture(scope="module")
def sphere():
    poly = pv.Sphere(radius=RADIUS, theta_resolution=40, phi_resolution=40)
    return poly_to_mesh(poly.triangulate())


@pytest.fixture(scope="module")
def jittered_sphere(sphere):
    """A sphere with its symmetry broken, for the tests that need one basis.

    A sphere's eigenvalues have multiplicity 2l+1, and eigenvectors inside a
    degenerate eigenspace are defined only up to a rotation of it -- two
    ARPACK runs on the same matrix return different ones. Any test that
    compares one decomposition against another needs a spectrum that is
    simple, so the radius is jittered.
    """
    vertices, faces = np.asarray(sphere[0]), np.asarray(sphere[1])
    rng = np.random.default_rng(3)
    scaling = 1.0 + 0.05 * rng.normal(size=(len(vertices), 1))
    return (vertices * scaling, faces)


#: The tube the cap-versus-mid-tube test is built from: a dendrite-scale
#: radius, long enough that the middle is many diffusion lengths from either
#: end at the largest timescale here.
TUBE_RADIUS = 400.0
TUBE_LENGTH = 12_000.0
SPACING = 100.0


@pytest.fixture(scope="module")
def capped_tube():
    """A tube closed by a hemisphere at the top and open at the bottom.

    The two cases the heat kernel signature conflates, in one mesh: near the
    closed end heat can only leave one way, and mid-tube it leaves both ways.
    """
    z, radius = tube_profile(TUBE_LENGTH, TUBE_RADIUS, SPACING)
    cap_z, cap_radius = hemisphere_profile(TUBE_RADIUS, z[-1], SPACING)
    return revolve_profile(
        np.concatenate([z, cap_z[1:]]),
        np.concatenate([radius, cap_radius[1:]]),
        n_theta=40,
    )


def test_signal_filter_matches_a_dense_heat_kernel(jittered_sphere):
    """K_t f accumulated band by band equals the dense kernel's action."""
    sphere = jittered_sphere
    L, M = cotangent_laplacian(sphere, robust=True)
    eigenvalues, eigenvectors = spectral_geometry_filter(
        (L, M), None, max_eigenvalue=MAX_EIGENVALUE, drop_first=False
    )

    rng = np.random.default_rng(0)
    signals = np.column_stack(
        [np.asarray(sphere[0])[:, 0], rng.normal(size=len(sphere[0]))]
    )

    filter_func = get_hks_filter(T_MAX, T_MIN, 4)
    _, filtered = spectral_geometry_filter(
        (L, M),
        filter_func,
        max_eigenvalue=MAX_EIGENVALUE,
        drop_first=False,
        signals=signals,
    )

    scales = np.geomspace(T_MIN, T_MAX, 4)
    weighted = M @ signals
    for index, scale in enumerate(scales):
        kernel = eigenvectors * np.exp(-eigenvalues * scale)
        expected = eigenvectors @ (kernel.T @ weighted)
        np.testing.assert_allclose(
            filtered[:, index, :],
            expected,
            rtol=1e-6,
            atol=1e-6 * np.abs(expected).max(),
        )


def test_the_measure_has_unit_mass(sphere):
    """(K_t 1)(x) is 1 however far the spectrum is truncated.

    Only the constant mode has nonzero <phi_k, 1>_M, so the truncation cannot
    touch it -- which is why the moments need no normalising, and why the
    constant mode must not be dropped.
    """
    L, M = cotangent_laplacian(sphere, robust=True)
    ones = np.ones((len(sphere[0]), 1))
    _, filtered = spectral_geometry_filter(
        (L, M),
        get_hks_filter(T_MAX, T_MIN, 4),
        max_eigenvalue=MAX_EIGENVALUE,
        drop_first=False,
        signals=ones,
    )
    np.testing.assert_allclose(filtered[:, :, 0], 1.0, rtol=1e-6)


def test_dropping_the_constant_mode_loses_the_mass(sphere):
    """The guard behind `drop_first=False`, stated as a measurement."""
    L, M = cotangent_laplacian(sphere, robust=True)
    ones = np.ones((len(sphere[0]), 1))
    _, filtered = spectral_geometry_filter(
        (L, M),
        get_hks_filter(T_MAX, T_MIN, 4),
        max_eigenvalue=MAX_EIGENVALUE,
        drop_first=True,
        signals=ones,
    )
    assert np.abs(filtered).max() < 1e-6


def test_signals_without_a_filter_are_refused(sphere):
    with pytest.raises(ValueError, match="signals need a filter"):
        spectral_geometry_filter(sphere, None, signals=np.ones((len(sphere[0]), 1)))


def test_signals_of_the_wrong_length_are_refused(sphere):
    with pytest.raises(ValueError, match="must be"):
        spectral_geometry_filter(
            sphere,
            get_hks_filter(T_MAX, T_MIN, 2),
            max_eigenvalue=MAX_EIGENVALUE,
            signals=np.ones((3, 1)),
        )


def _moments(mesh, n_scales=4):
    return compute_heat_kernel_moments(
        mesh,
        max_eigenvalue=MAX_EIGENVALUE,
        t_min=T_MIN,
        t_max=T_MAX,
        n_scales=n_scales,
        robust=True,
    )


def _columns(moments: np.ndarray, n_scales: int = 4) -> dict[str, np.ndarray]:
    """The moment array as named columns, so a test can say what it means."""
    names = heat_kernel_moment_names(n_scales)
    return {name: moments[:, index] for index, name in enumerate(names)}


def test_the_column_count_matches_the_names(sphere):
    moments = _moments(sphere, n_scales=3)
    assert moments.shape == (len(sphere[0]), 3 * len(HEAT_KERNEL_MOMENT_NAMES))
    assert len(heat_kernel_moment_names(3)) == moments.shape[1]


def test_moments_are_translation_invariant(sphere):
    vertices, faces = np.asarray(sphere[0]), np.asarray(sphere[1])
    moved = (vertices + np.array([5e5, -2e5, 3e5]), faces)
    np.testing.assert_allclose(_moments(sphere), _moments(moved), rtol=1e-3, atol=1e-3)


def test_moments_are_rotation_invariant(sphere):
    vertices, faces = np.asarray(sphere[0]), np.asarray(sphere[1])
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    turned = (vertices @ rotation.T, faces)
    np.testing.assert_allclose(_moments(sphere), _moments(turned), rtol=1e-3, atol=1e-3)


def test_vertex_normals_are_unit_and_radial_on_a_sphere(sphere):
    normals = vertex_normals(sphere)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=1e-10)
    radial = np.asarray(sphere[0]) / RADIUS
    cosines = (normals * radial).sum(axis=1)
    # Sign is the file's, magnitude is the geometry's.
    assert np.abs(cosines).min() > 0.99


def test_a_sphere_drifts_inward_and_sees_a_flat_neighbourhood(sphere):
    """Everything a sphere's moments say is fixed by symmetry.

    Heat smoothing is mean-curvature flow, so the drift is radially inward
    everywhere; the neighbourhood is a geodesic disc, so its covariance has
    two large eigenvalues and one small one -- planar, not linear.
    """
    columns = _columns(_moments(sphere))

    assert np.median(columns["normal_0"]) < 0
    assert (columns["normal_0"] < 0).mean() > 0.99
    np.testing.assert_allclose(columns["drift_0"], -columns["normal_0"], rtol=1e-2)
    # Purely normal by symmetry: nothing on a sphere is one-sided. The floor
    # is the triangulation's, not the descriptor's -- pv.Sphere's rings are
    # not uniform near the poles.
    assert np.median(columns["tangent_0"]) < 1e-2 * np.median(columns["drift_0"])

    # And the magnitude has a closed form too: the coordinates are the l=1
    # spherical harmonics, eigenvalue 2/R^2, so the drift is
    # R(1 - exp(-2t/R^2)) everywhere on the sphere.
    expected = RADIUS * (1 - np.exp(-2 * T_MIN / RADIUS**2))
    np.testing.assert_allclose(np.median(columns["drift_0"]), expected, rtol=1e-2)
    assert np.median(columns["linear_0"]) < 0.1
    assert np.median(columns["planar_0"]) > 0.7


def test_a_tube_drifts_purely_inward(capped_tube):
    """Mid-tube the drift is the cylinder's own mean curvature and nothing else.

    This is why ``drift`` alone cannot find a cap: on a tube of radius R the
    drift is inward, normal, and t/(2R) in magnitude, which at spine scale on
    a dendrite-radius tube is hundreds of nanometres. A cap's drift has to
    beat that to be visible in the magnitude, and it does not have to.
    """
    axis = np.asarray(capped_tube[0])[:, 2]
    mid_tube = np.abs(axis) < TUBE_LENGTH / 6
    assert mid_tube.sum() > 200

    columns = _columns(_moments(capped_tube))
    normal = np.median(columns["normal_3"][mid_tube])
    tangent = np.median(columns["tangent_3"][mid_tube])

    # Exactly, not approximately. The transverse coordinates on a cylinder of
    # radius R are its angular n=1 eigenfunctions, with eigenvalue 1/R^2, so
    # K_t x = exp(-t/R^2) x and the inward drift is R(1 - exp(-t/R^2)). This
    # is the end-to-end check on the whole construction: Laplacian,
    # band-by-band solve, signal projection, and the drift assembly all have
    # to be right for a closed form to come out.
    assert normal < 0
    expected = TUBE_RADIUS * (1 - np.exp(-T_MAX / TUBE_RADIUS**2))
    np.testing.assert_allclose(-normal, expected, rtol=1e-3)
    # Six orders of magnitude down, not merely small.
    assert tangent < 1e-4 * abs(normal)


def test_a_cap_gives_the_drift_a_tangential_part(capped_tube):
    """The distinction the heat kernel signature is blind to, on one mesh.

    Mid-tube heat leaves symmetrically along the axis, so the drift has no
    tangential part. Below the cap's rim it can leave only one way, so it
    does. At the pole symmetry is restored and the tangential part collapses
    again -- but the normal part rises, because a hemisphere's mean curvature
    is twice a cylinder's of the same radius. The two columns cover the cap
    between them, which is why both are emitted.
    """
    axis = np.asarray(capped_tube[0])[:, 2]
    rim = axis.max() - TUBE_RADIUS
    mid_tube = np.abs(axis) < TUBE_LENGTH / 6
    shoulder = (axis > rim - 1.5 * TUBE_RADIUS) & (axis < rim)
    pole = axis > axis.max() - 0.3 * TUBE_RADIUS
    assert shoulder.sum() > 50 and pole.sum() > 50

    columns = _columns(_moments(capped_tube))
    assert np.median(columns["tangent_3"][shoulder]) > 1000 * np.median(
        columns["tangent_3"][mid_tube]
    )
    assert abs(np.median(columns["normal_3"][pole])) > 1.5 * abs(
        np.median(columns["normal_3"][mid_tube])
    )
    assert np.median(columns["linear_3"][mid_tube]) > np.median(
        columns["linear_3"][pole]
    )
