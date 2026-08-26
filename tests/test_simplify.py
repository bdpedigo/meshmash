import numpy as np
import pytest

from meshmash import simplify_mesh


@pytest.fixture(scope="module")
def simplified(mesh):
    return simplify_mesh(mesh, target_reduction=0.7)


def test_simplify_mesh_reduces_the_face_count(mesh, simplified):
    (_, faces), _ = simplified
    assert faces.shape[0] < mesh[1].shape[0]


def test_simplify_mesh_mapping_covers_every_input_vertex(mesh, simplified):
    _, mapping = simplified
    assert mapping.shape == (mesh[0].shape[0],)


def test_simplify_mesh_mapping_indexes_the_simplified_mesh(simplified):
    (vertices, _), mapping = simplified
    assert mapping.min() >= 0
    assert mapping.max() < vertices.shape[0]


def test_simplify_mesh_mapping_agrees_with_the_returned_faces(simplified):
    """The reason both come from replay: the other ordering does not agree.

    Remapping the input's own faces through the mapping has to land on real
    vertices of the mesh that came back, which is exactly what fails if the
    mesh is taken from ``simplify`` and the mapping from
    ``replay_simplification``.
    """
    (vertices, faces), mapping = simplified
    assert faces.max() < vertices.shape[0]
    assert set(np.unique(faces)).issubset(set(np.unique(mapping)))


def test_simplify_mesh_none_is_a_pass_through(mesh):
    (vertices, faces), mapping = simplify_mesh(mesh, target_reduction=None)
    np.testing.assert_array_equal(vertices, mesh[0])
    np.testing.assert_array_equal(faces, mesh[1])
    np.testing.assert_array_equal(mapping, np.arange(mesh[0].shape[0]))


def test_simplify_mesh_reduction_controls_how_much_is_removed(mesh):
    (_, light), _ = simplify_mesh(mesh, target_reduction=0.2)
    (_, heavy), _ = simplify_mesh(mesh, target_reduction=0.9)
    assert heavy.shape[0] < light.shape[0] < mesh[1].shape[0]
