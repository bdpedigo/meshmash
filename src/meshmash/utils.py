import numpy as np
import pandas as pd
import pyvista as pv
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from .types import Mesh, interpret_mesh


def mesh_to_poly(mesh: Mesh) -> pv.PolyData:
    if isinstance(mesh, pv.PolyData):
        return mesh
    elif isinstance(mesh, tuple):
        return pv.make_tri_mesh(*mesh)
    elif hasattr(mesh, "polydata"):
        return mesh.polydata
    elif hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        return pv.make_tri_mesh(mesh.vertices, mesh.faces)
    else:
        raise ValueError("Invalid mesh input.")


def mesh_to_edges(mesh: Mesh) -> np.ndarray:
    poly = mesh_to_poly(mesh)
    edge_data = poly.extract_all_edges(use_all_points=True, clear_data=True)
    lines = edge_data.lines
    edges = lines.reshape(-1, 3)[:, 1:]
    return edges


def mesh_to_adjacency(mesh: Mesh) -> csr_array:
    # TODO only use here because this is faster than numpy unique for unique extracting
    # edges, should be some other way to do this
    poly = mesh_to_poly(mesh)
    edge_data = poly.extract_all_edges(use_all_points=True, clear_data=True)
    lines = edge_data.lines
    edges = lines.reshape(-1, 3)[:, 1:]
    vertices = poly.points
    n_vertices = len(vertices)

    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)

    # was having some issue here with dijkstra, so casting to intc
    # REF: https://github.com/scipy/scipy/issues/20904
    # REF: related https://github.com/scipy/scipy/issues/20817
    adj = csr_array(
        (edge_lengths, (edges[:, 0].astype(np.intc), edges[:, 1].astype(np.intc))),
        shape=(n_vertices, n_vertices),
    )

    return adj


def poly_to_mesh(poly: pv.PolyData) -> Mesh:
    vertices = np.asarray(poly.points)
    faces = poly.faces.reshape(-1, 4)[:, 1:]
    return vertices, faces


def fix_mesh(mesh, **kwargs):
    from pymeshfix import MeshFix

    poly = mesh_to_poly(mesh)
    mesh_fix = MeshFix(poly)
    mesh_fix.repair(**kwargs)
    return poly_to_mesh(mesh_fix.mesh)


def project_points_to_mesh(
    points, mesh, distance_threshold=None, return_distances=False
):
    if isinstance(mesh, tuple):
        vertices = mesh[0]
    else:
        vertices = mesh.vertices
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(vertices)

    distances, indices = nn.kneighbors(points)
    indices = indices.reshape(-1)
    distances = distances.reshape(-1)
    if distance_threshold is not None:
        indices[distances > distance_threshold] = -1

    if return_distances:
        return indices, distances
    else:
        return indices


def component_size_transform(mesh, indices=None):
    """Returns the size of each connected component in a mesh."""
    mesh = interpret_mesh(mesh)
    if indices is None:
        indices = np.arange(len(mesh[0]))
    adj = mesh_to_adjacency(mesh)
    _, component_labels = connected_components(adj, directed=False)

    unique_labels, counts = np.unique(component_labels, return_counts=True)
    size_map = dict(zip(unique_labels, counts))

    index_components = component_labels[indices]
    component_sizes = np.array([size_map[label] for label in index_components])

    return component_sizes


def get_label_components(mesh, labels):
    """Returns the connected components of a mesh which share the same label."""
    if isinstance(labels, pd.Series):
        labels = labels.values

    mesh = interpret_mesh(mesh)
    edges = mesh_to_edges(mesh)
    source_labels = labels[edges[:, 0]]
    target_labels = labels[edges[:, 1]]

    edges = edges[source_labels == target_labels]

    clipped_graph = csr_array(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(mesh[0].shape[0], mesh[0].shape[0]),
    )

    _, component_labels = connected_components(clipped_graph, directed=False)

    return component_labels


def subset_mesh_by_indices(mesh: Mesh, indices: np.ndarray) -> Mesh:
    vertices, faces = interpret_mesh(mesh)
    new_vertices = vertices[indices]
    index_mapping = dict(zip(indices, np.arange(len(indices))))
    # use numpy to get faces for which all indices are in the subset
    face_mask = np.all(np.isin(faces, indices), axis=1)
    new_faces = np.vectorize(index_mapping.get)(faces[face_mask])
    return new_vertices, new_faces


def largest_mesh_component(mesh: Mesh) -> Mesh:
    adj = mesh_to_adjacency(mesh)
    _, component_labels = connected_components(adj, directed=False)
    largest_component = np.argmax(np.bincount(component_labels))
    indices = np.where(component_labels == largest_component)[0]
    return subset_mesh_by_indices(mesh, indices)
