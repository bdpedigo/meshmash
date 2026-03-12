import numpy as np

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
