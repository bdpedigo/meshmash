"""The spectral filter's algebra: the kernel diagonal, and its action on signals.

The band-by-band accumulation is linear algebra with a right answer, and a
dense heat kernel built from the same eigenpairs gives that answer exactly.
These check against it rather than against recorded numbers.
"""

import numpy as np
import pytest
import pyvista as pv

from meshmash import (
    concatenate_filters,
    get_heat_filter,
    get_hks_filter,
    spectral_geometry_filter,
)
from meshmash.decompose import filter_width
from meshmash.laplacian import cotangent_laplacian
from meshmash.utils import poly_to_mesh

#: A sphere this size in nm puts its eigenvalues, l(l+1)/R^2, in the range a
#: dendrite chunk's sit in, so these timescales are the ones a real featurizer
#: uses rather than numbers invented for a unit sphere.
RADIUS = 1000.0
MAX_EIGENVALUE = 1e-4
T_MIN, T_MAX = 1e4, 2.5e5


@pytest.fixture(scope="module")
def sphere():
    poly = pv.Sphere(radius=RADIUS, theta_resolution=40, phi_resolution=40)
    return poly_to_mesh(poly.triangulate())


@pytest.fixture(scope="module")
def jittered_sphere(sphere):
    """A sphere with its symmetry broken, for tests that need one fixed basis.

    A sphere's eigenvalues have multiplicity 2l+1, and eigenvectors inside a
    degenerate eigenspace are defined only up to a rotation of it, so two
    ARPACK runs on the same matrix return different ones. Any test that
    compares one decomposition against another needs a simple spectrum, so the
    radius is jittered.
    """
    vertices, faces = np.asarray(sphere[0]), np.asarray(sphere[1])
    rng = np.random.default_rng(3)
    return (vertices * (1.0 + 0.05 * rng.normal(size=(len(vertices), 1))), faces)


def test_get_heat_filter_matches_get_hks_filter_on_a_geometric_grid():
    """`get_hks_filter` is `get_heat_filter` over a geomspace, and stays so."""
    eigenvalues = np.array([0.0, 1e-7, 1e-6, 1e-5])
    expected = get_hks_filter(T_MAX, T_MIN, 8)(eigenvalues)
    got = get_heat_filter(np.geomspace(T_MIN, T_MAX, 8))(eigenvalues)
    np.testing.assert_array_equal(got, expected)


def test_concatenate_filters_stacks_the_banks_in_order():
    eigenvalues = np.array([0.0, 1e-6])
    first = get_heat_filter([1e4, 1e5, 1e6])
    second = get_heat_filter([1e7])
    stacked = concatenate_filters(first, second)(eigenvalues)
    assert stacked.shape == (4, 2)
    np.testing.assert_array_equal(stacked[:3], first(eigenvalues))
    np.testing.assert_array_equal(stacked[3:], second(eigenvalues))


def test_concatenate_filters_skips_none():
    """An optional filter passes through with no branch at the call site."""
    eigenvalues = np.array([0.0, 1e-6])
    only = get_heat_filter([1e5])
    np.testing.assert_array_equal(
        concatenate_filters(None, only)(eigenvalues), only(eigenvalues)
    )
    with pytest.raises(ValueError, match="at least one filter"):
        concatenate_filters(None, None)


def test_filter_width_counts_the_output_rows():
    assert filter_width(get_hks_filter(T_MAX, T_MIN, 8)) == 8
    bank = concatenate_filters(get_heat_filter([1e4, 1e5]), get_heat_filter([1e6]))
    assert filter_width(bank) == 3


def test_signal_filter_matches_a_dense_heat_kernel(jittered_sphere):
    """K_t f accumulated band by band equals the dense kernel's action."""
    L, M = cotangent_laplacian(jittered_sphere, robust=True)
    eigenvalues, eigenvectors = spectral_geometry_filter(
        (L, M), None, max_eigenvalue=MAX_EIGENVALUE, drop_first=False
    )

    rng = np.random.default_rng(0)
    vertices = np.asarray(jittered_sphere[0])
    signals = np.column_stack([vertices[:, 0], rng.normal(size=len(vertices))])

    _, filtered = spectral_geometry_filter(
        (L, M),
        get_hks_filter(T_MAX, T_MIN, 4),
        max_eigenvalue=MAX_EIGENVALUE,
        drop_first=False,
        signals=signals,
    )

    weighted = M @ signals
    for index, scale in enumerate(np.geomspace(T_MIN, T_MAX, 4)):
        kernel = eigenvectors * np.exp(-eigenvalues * scale)
        expected = eigenvectors @ (kernel.T @ weighted)
        np.testing.assert_allclose(
            filtered[:, index, :],
            expected,
            rtol=1e-6,
            atol=1e-6 * np.abs(expected).max(),
        )


def test_the_diagonal_path_is_unchanged_by_asking_for_signals(jittered_sphere):
    """Signals ride alongside the diagonal, they do not perturb it."""
    L, M = cotangent_laplacian(jittered_sphere, robust=True)
    kwargs = dict(max_eigenvalue=MAX_EIGENVALUE, drop_first=True)
    alone = spectral_geometry_filter((L, M), get_hks_filter(T_MAX, T_MIN, 4), **kwargs)
    together, _ = spectral_geometry_filter(
        (L, M),
        get_hks_filter(T_MAX, T_MIN, 4),
        signals=np.ones((L.shape[0], 1)),
        **kwargs,
    )
    np.testing.assert_allclose(together, alone, rtol=1e-10)


def test_the_measure_has_unit_mass(sphere):
    """(K_t 1)(x) is 1 however far the spectrum is truncated.

    Only the constant mode has nonzero <phi_k, 1>_M, so truncation cannot touch
    it, which is why the constant mode must not be dropped from a signal.
    """
    L, M = cotangent_laplacian(sphere, robust=True)
    _, filtered = spectral_geometry_filter(
        (L, M),
        get_hks_filter(T_MAX, T_MIN, 4),
        max_eigenvalue=MAX_EIGENVALUE,
        drop_first=False,
        signals=np.ones((len(sphere[0]), 1)),
    )
    np.testing.assert_allclose(filtered[:, :, 0], 1.0, rtol=1e-6)


def test_dropping_the_constant_mode_loses_the_mass(sphere):
    """The guard behind `drop_first=False`, stated as a measurement."""
    L, M = cotangent_laplacian(sphere, robust=True)
    _, filtered = spectral_geometry_filter(
        (L, M),
        get_hks_filter(T_MAX, T_MIN, 4),
        max_eigenvalue=MAX_EIGENVALUE,
        drop_first=True,
        signals=np.ones((len(sphere[0]), 1)),
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
