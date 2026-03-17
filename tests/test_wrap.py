import numpy as np

from meshmash import wrap_mesh


def test_wrap_mesh_returns_mesh_tuple(mesh):
    vertices, faces = wrap_mesh(mesh)
    assert isinstance(vertices, np.ndarray)
    assert isinstance(faces, np.ndarray)
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3


def test_wrap_mesh_preserves_dtype(mesh):
    vertices, faces = wrap_mesh(mesh)
    assert vertices.dtype == mesh[0].dtype
    assert faces.dtype == mesh[1].dtype


def test_wrap_mesh_alpha_fraction(mesh):
    vertices, faces = wrap_mesh(mesh, alpha=None, offset=None, alpha_fraction=0.02, offset_fraction=0.002)
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
