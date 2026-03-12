import numpy as np
import pyvista as pv

from meshmash.utils import (
    combine_meshes,
    component_size_transform,
    compute_distances_to_point,
    largest_mesh_component,
    mesh_to_adjacency,
    mesh_to_edges,
    mesh_to_poly,
    subset_mesh_by_indices,
)


def test_mesh_to_poly(mesh):
    poly = mesh_to_poly(mesh)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_points == mesh[0].shape[0]


def test_mesh_to_edges_shape(mesh):
    edges = mesh_to_edges(mesh)
    assert edges.ndim == 2
    assert edges.shape[1] == 2


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
