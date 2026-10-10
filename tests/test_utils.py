import numpy as np
import pytest
import pyvista as pv

import meshmash.utils
from meshmash.utils import (
    combine_meshes,
    component_size_transform,
    compute_distances_to_point,
    largest_mesh_component,
    mesh_to_adjacency,
    mesh_to_edges,
    mesh_to_poly,
    remove_repeated_vertex_faces,
    rough_subset_mesh_by_indices,
    subset_mesh_by_indices,
)


@pytest.mark.parametrize("face_dtype", [np.uint32, np.int32, np.int64])
def test_rough_subset_mesh_by_indices(mesh, face_dtype):
    vertices, faces = mesh
    faces = faces.astype(face_dtype)
    seeds = np.arange(0, len(vertices), 7)

    (new_vertices, new_faces), vertex_indices = rough_subset_mesh_by_indices(
        (vertices, faces), seeds
    )

    assert new_faces.dtype == face_dtype
    face_mask = np.any(np.isin(faces, seeds), axis=1)
    np.testing.assert_array_equal(new_vertices[new_faces], vertices[faces[face_mask]])
    np.testing.assert_array_equal(new_vertices, vertices[vertex_indices])


def test_remove_repeated_vertex_faces():
    vertices = np.arange(15, dtype=float).reshape(5, 3)
    faces = np.array([[0, 1, 2], [1, 1, 3], [2, 3, 4], [4, 4, 4], [3, 0, 3]])

    kept_vertices, kept_faces = remove_repeated_vertex_faces((vertices, faces))

    assert kept_vertices is vertices
    np.testing.assert_array_equal(kept_faces, [[0, 1, 2], [2, 3, 4]])
    again_vertices, again_faces = remove_repeated_vertex_faces(
        (kept_vertices, kept_faces)
    )
    assert again_vertices is vertices
    np.testing.assert_array_equal(again_faces, kept_faces)


def test_mesh_to_poly(mesh):
    poly = mesh_to_poly(mesh)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_points == mesh[0].shape[0]


def test_mesh_to_edges_shape(mesh):
    edges = mesh_to_edges(mesh)
    assert edges.ndim == 2
    assert edges.shape[1] == 2


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64, np.uint64])
def test_mesh_to_edges_are_the_faces_edges_once_each(mesh, dtype):
    """uint32 faces are what CloudVolume returns, and an older VTK path got them wrong."""
    vertices, faces = mesh
    faces = np.asarray(faces).astype(dtype)
    corners = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    expected = np.unique(np.sort(corners, axis=1), axis=0)

    edges = mesh_to_edges((vertices, faces))

    np.testing.assert_array_equal(edges, expected)


def test_mesh_to_edges_refuses_a_mesh_too_large_to_pack(mesh, monkeypatch):
    monkeypatch.setattr(meshmash.utils, "MAX_PACKED_VERTICES", len(mesh[0]) - 1)
    with pytest.raises(ValueError, match="int64"):
        mesh_to_edges(mesh)


@pytest.mark.parametrize("vertex_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("face_dtype", [np.int32, np.uint32, np.int64, np.uint64])
def test_mesh_to_adjacency_holds_each_edge_once_at_its_length(
    mesh, vertex_dtype, face_dtype
):
    vertices = np.asarray(mesh[0]).astype(vertex_dtype)
    faces = np.asarray(mesh[1]).astype(face_dtype)

    adjacency = mesh_to_adjacency((vertices, faces)).tocoo()

    edges = mesh_to_edges(mesh)
    assert adjacency.nnz == len(edges)
    np.testing.assert_array_equal(
        adjacency.row < adjacency.col, True
    )  # upper triangular
    lengths = np.linalg.norm(vertices[adjacency.row] - vertices[adjacency.col], axis=1)
    np.testing.assert_allclose(adjacency.data, lengths, rtol=1e-6)
    assert adjacency.data.dtype == vertex_dtype


def test_mesh_to_adjacency_shape(mesh):
    n = mesh[0].shape[0]
    adj = mesh_to_adjacency(mesh)
    assert adj.shape == (n, n)
    assert np.all(adj.data > 0)


def test_subset_mesh_by_indices(mesh):
    k = 100
    indices = np.arange(k)
    sub_vertices, sub_faces = subset_mesh_by_indices(mesh, indices)
    assert sub_vertices.shape[0] <= k
    assert sub_vertices.shape[1] == 3


def test_largest_mesh_component(mesh):
    vertices, faces = largest_mesh_component(mesh)
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3


def test_compute_distances_to_point(mesh):
    vertices = mesh[0]
    center = vertices.mean(axis=0)
    dists = compute_distances_to_point(vertices, center)
    assert len(dists) == len(vertices)
    assert np.all(dists >= 0)


def test_combine_meshes(mesh):
    n_original = mesh[0].shape[0]
    combined_vertices, _ = combine_meshes([mesh, mesh])
    assert combined_vertices.shape[0] == 2 * n_original


def test_component_size_transform(mesh):
    n = mesh[0].shape[0]
    sizes = component_size_transform(mesh)
    assert len(sizes) == n
    assert np.all(sizes >= 1)
