import pytest

from meshmash import fetch_sample_mesh


def test_fetch_sample_mesh(mesh):
    vertices, faces = mesh
    assert vertices.ndim == 2
    assert vertices.shape[1] == 3
    assert faces.ndim == 2
    assert faces.shape[1] == 3


def test_fetch_sample_mesh_unknown_name():
    with pytest.raises(ValueError):
        fetch_sample_mesh("nonexistent_mesh")
