import numpy as np
import pandas as pd
import pytest

from meshmash.agglomerate import aggregate_features
from meshmash.laplacian import area_matrix, compute_vertex_areas, cotangent_laplacian


def test_area_matrix_shape(mesh):
    n = mesh[0].shape[0]
    M = area_matrix(mesh)
    assert M.shape == (n, n)


def test_area_matrix_positive_diag(mesh):
    M = area_matrix(mesh)
    assert np.all(M.diagonal() > 0)


def test_cotangent_laplacian_returns_pair(mesh):
    result = cotangent_laplacian(mesh)
    assert len(result) == 2


def test_cotangent_laplacian_shape(mesh):
    n = mesh[0].shape[0]
    L, M = cotangent_laplacian(mesh)
    assert L.shape == (n, n)
    assert M.shape == (n, n)


def test_cotangent_laplacian_symmetry(mesh):
    L, _ = cotangent_laplacian(mesh)
    residual = L - L.T
    if residual.nnz > 0:
        non_nan = residual.data[~np.isnan(residual.data)]
        if len(non_nan) > 0:
            assert np.max(np.abs(non_nan)) < 1e-10


def test_compute_vertex_areas_length(mesh):
    n = mesh[0].shape[0]
    areas = compute_vertex_areas(mesh)
    assert len(areas) == n


def test_compute_vertex_areas_positive(mesh):
    areas = compute_vertex_areas(mesh)
    assert np.all(areas > 0)


def test_compute_vertex_areas_defaults_to_the_robust_mass_matrix(mesh):
    # The spectral operators and curvature measures default to robust, so the
    # aggregation weights must too.
    _, M = cotangent_laplacian(mesh, robust=True)
    np.testing.assert_array_equal(compute_vertex_areas(mesh), M.diagonal())


def _degenerate_mesh():
    """A fan of three good faces plus a zero-area face, a duplicated vertex,
    and a vertex that no face uses."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.5],
            [2.0, 0.0, 0.0],  # collinear with 0 and 1: zero-area face
            [1.0, 0.0, 0.0],  # duplicates vertex 1
            [5.0, 5.0, 5.0],  # in no face
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 5], [0, 1, 4]])
    return vertices, faces


@pytest.mark.parametrize("robust", [True, False])
def test_vertex_areas_are_finite_and_non_negative_on_a_degenerate_mesh(robust):
    areas = compute_vertex_areas(_degenerate_mesh(), robust=robust)
    assert np.isfinite(areas).all()
    assert (areas >= 0).all()
    # Mollification gives even the unused vertex a tiny positive area.
    assert (areas[6] > 0) if robust else (areas[6] == 0)


def test_a_domain_of_zero_area_vertices_aggregates_to_a_finite_mean():
    features = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
    labels = np.array([0, 0, 1, 1])
    weights = np.array([1.0, 3.0, 0.0, 0.0])
    out = aggregate_features(features, labels, func="mean", weights=weights)
    assert out.loc[0, "f"] == pytest.approx(1.75)
    assert out.loc[1, "f"] == pytest.approx(3.5)  # unweighted fallback


def test_aggregate_features_does_not_depend_on_neighbouring_column_dtypes():
    # pandas used to keep float64 for a mixed frame and round an all-float32
    # one, so the same column aggregated differently depending on its company.
    rng = np.random.default_rng(0)
    shared = rng.random(100).astype(np.float32)
    labels = rng.integers(0, 7, size=100)
    weights = rng.random(100)
    alone = pd.DataFrame({"f": shared})
    mixed = pd.DataFrame({"f": shared, "g": rng.random(100)})

    alone_out = aggregate_features(alone, labels, weights=weights)
    mixed_out = aggregate_features(mixed, labels, weights=weights)

    assert (alone_out.dtypes == np.float32).all()
    assert (mixed_out.dtypes == np.float32).all()
    assert alone_out.index.dtype == np.int32
    pd.testing.assert_series_equal(alone_out["f"], mixed_out["f"])
