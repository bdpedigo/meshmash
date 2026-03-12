from typing import Generator, Literal, Optional, Union, overload

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from .types import ArrayLike, Mesh, interpret_mesh


def mesh_to_poly(mesh: Mesh) -> pv.PolyData:
    """Convert a mesh to a :class:`pyvista.PolyData` object.

    Parameters
    ----------
    mesh :
        Input mesh.  Accepts a ``(vertices, faces)`` tuple, a
        :class:`pyvista.PolyData`, or any object with ``vertices`` and
        ``faces`` attributes.

    Returns
    -------
    :
        Triangle mesh as a :class:`pyvista.PolyData`.
    """
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
    """Extract all edges from a mesh as vertex index pairs.

    Parameters
    ----------
    mesh :
        Input mesh accepted by :func:`mesh_to_poly`.

    Returns
    -------
    :
        Array of edge vertex index pairs, shape ``(E, 2)``.
    """
    poly = mesh_to_poly(mesh)
    edge_data = poly.extract_all_edges(use_all_points=True, clear_data=True)
    lines = edge_data.lines
    edges = lines.reshape(-1, 3)[:, 1:]
    return edges


def mesh_to_adjacency(mesh: Mesh) -> csr_array:
    """Build a sparse weighted adjacency matrix from a mesh.

    Edge weights are Euclidean edge lengths.  The returned matrix is upper
    triangular (each undirected edge appears once).

    Parameters
    ----------
    mesh :
        Input mesh accepted by :func:`mesh_to_poly`.

    Returns
    -------
    :
        Sparse CSR adjacency matrix of shape ``(V, V)`` with edge-length
        weights.
    """
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
    """Convert a :class:`pyvista.PolyData` to a ``(vertices, faces)`` tuple.

    Parameters
    ----------
    poly :
        Triangle surface mesh as a :class:`pyvista.PolyData`.

    Returns
    -------
    vertices :
        Array of vertex positions, shape ``(V, 3)``.
    faces :
        Array of triangle face indices, shape ``(F, 3)``.
    """
    vertices = np.asarray(poly.points)
    faces = poly.faces.reshape(-1, 4)[:, 1:]
    return vertices, faces


def fix_mesh(mesh: Mesh, **kwargs) -> Mesh:
    """Repair a mesh using :mod:`pymeshfix`.

    Attempts to close holes and remove self-intersections so that the
    result is a manifold surface.  Additional keyword arguments are
    forwarded to :meth:`pymeshfix.MeshFix.repair`.

    Parameters
    ----------
    mesh :
        Input mesh accepted by :func:`mesh_to_poly`.
    **kwargs :
        Keyword arguments forwarded to :meth:`pymeshfix.MeshFix.repair`.

    Returns
    -------
    vertices :
        Array of repaired vertex positions, shape ``(V', 3)``.
    faces :
        Array of repaired triangle face indices, shape ``(F', 3)``.
    """
    from pymeshfix import MeshFix

    poly = mesh_to_poly(mesh)
    mesh_fix = MeshFix(poly)
    mesh_fix.repair(**kwargs)
    return poly_to_mesh(mesh_fix.mesh)


@overload
def project_points_to_mesh(
    points: ArrayLike,
    mesh: Mesh,
    distance_threshold: Optional[float] = None,
    return_distances: Literal[False] = ...,
) -> np.ndarray: ...


@overload
def project_points_to_mesh(
    points: ArrayLike,
    mesh: Mesh,
    distance_threshold: Optional[float] = None,
    *,
    return_distances: Literal[True],
) -> tuple[np.ndarray, np.ndarray]: ...


def project_points_to_mesh(
    points: ArrayLike,
    mesh: Mesh,
    distance_threshold: Optional[float] = None,
    return_distances: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Find the nearest mesh vertex for each query point.

    Parameters
    ----------
    points :
        Query point coordinates, shape ``(N, 3)``.
    mesh :
        Target mesh accepted by :func:`mesh_to_poly`.
    distance_threshold :
        If provided, query points whose nearest vertex is farther than this
        distance are assigned an index of ``-1``.
    return_distances :
        If ``True``, also return the distance to the nearest vertex for each
        query point.

    Returns
    -------
    indices :
        Indices of the nearest vertex in the mesh for each query point,
        shape ``(N,)``.  Entries are ``-1`` where the nearest vertex
        exceeds ``distance_threshold``.
    distances :
        Only returned when ``return_distances=True``.  Euclidean distances
        to the nearest vertex, shape ``(N,)``.
    """
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


def component_size_transform(
    mesh: Mesh, indices: Optional[np.ndarray] = None
) -> np.ndarray:
    """Return the connected-component size for each vertex.

    Parameters
    ----------
    mesh :
        Input mesh accepted by :func:`mesh_to_poly`.
    indices :
        Subset of vertex indices to return sizes for.  If ``None``, sizes
        are returned for all vertices.

    Returns
    -------
    :
        Array of component sizes (number of vertices in the same connected
        component), one value per entry in ``indices`` (or per vertex if
        ``indices`` is ``None``).
    """
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


def get_label_components(mesh: Mesh, labels: ArrayLike) -> np.ndarray:
    """Label connected components of a mesh that share the same vertex label.

    Two vertices belong to the same component only if they are connected by
    an edge *and* carry the same label value.

    Parameters
    ----------
    mesh :
        Input mesh accepted by :func:`mesh_to_poly`.
    labels :
        Per-vertex label array of length ``V``.

    Returns
    -------
    :
        Per-vertex component label array of length ``V``.  Vertices with
        different vertex labels will never share a component label.
    """
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
    """Extract a submesh containing only the specified vertices.

    Only faces whose *all three* vertices are in ``indices`` are kept.
    Face indices are remapped to the new, compact vertex numbering.

    Parameters
    ----------
    mesh :
        Input mesh accepted by :func:`interpret_mesh`.
    indices :
        Vertex indices to keep, or a boolean mask of length ``V``.

    Returns
    -------
    vertices :
        Subset of vertex positions, shape ``(len(indices), 3)``.
    faces :
        Remapped face index array containing only retained faces.
    """
    if indices.dtype == bool:
        indices = np.where(indices)[0]

    vertices, faces = interpret_mesh(mesh)
    new_vertices = vertices[indices]
    index_mapping = dict(zip(indices, np.arange(len(indices))))

    # use numpy to get faces for which all indices are in the subset
    face_mask = np.all(np.isin(faces, indices), axis=1)

    new_faces = np.vectorize(index_mapping.get, otypes=[faces.dtype])(faces[face_mask])
    return new_vertices, new_faces


def rough_subset_mesh_by_indices(
    mesh: Mesh, indices: np.ndarray
) -> tuple[Mesh, np.ndarray]:
    """Extract a submesh keeping faces where *any* vertex is in ``indices``.

    Unlike :func:`subset_mesh_by_indices`, a face is retained whenever at
    least one of its vertices is in ``indices``.  This can introduce
    additional vertices beyond those in ``indices``.

    Parameters
    ----------
    mesh :
        Input mesh accepted by :func:`interpret_mesh`.
    indices :
        Seed vertex indices.

    Returns
    -------
    submesh :
        ``(vertices, faces)`` tuple for the extracted submesh.
    vertex_indices :
        Final set of vertex indices (in the original mesh) that were
        included, which may be a superset of ``indices``.
    """
    vertices, faces = interpret_mesh(mesh)

    face_mask = np.any(np.isin(faces, indices), axis=1)
    vertex_indices = np.unique(faces[face_mask])
    new_vertices = vertices[vertex_indices]
    index_mapping = dict(zip(vertex_indices, np.arange(len(vertex_indices))))

    new_faces = np.vectorize(index_mapping.get)(faces[face_mask])
    new_mesh = (new_vertices, new_faces)
    return new_mesh, vertex_indices


def largest_mesh_component(mesh: Mesh) -> Mesh:
    """Return the largest connected component of a mesh.

    Parameters
    ----------
    mesh :
        Input mesh.

    Returns
    -------
    vertices :
        Vertices of the largest component.
    faces :
        Faces of the largest component, with remapped indices.
    """
    adj = mesh_to_adjacency(mesh)
    _, component_labels = connected_components(adj, directed=False)
    largest_component = np.argmax(np.bincount(component_labels))
    indices = np.where(component_labels == largest_component)[0]
    return subset_mesh_by_indices(mesh, indices)


def shuffle_label_mapping(x: ArrayLike) -> np.ndarray:
    """Randomly permute integer labels while preserving the set of label values.

    Useful for randomising colours when visualising label arrays.

    Parameters
    ----------
    x :
        Array of integer labels.

    Returns
    -------
    :
        Array of the same shape as ``x`` with label integers randomly
        permuted.
    """
    uni_labels = np.unique(x)
    new_labels = np.random.permutation(uni_labels)
    label_map = dict(zip(uni_labels, new_labels))
    x = np.array([label_map[label] for label in x])
    return x


def compute_distances_to_point(
    points: ArrayLike, center_point: ArrayLike
) -> np.ndarray:
    """Compute the Euclidean distance from each point to a reference point.

    Parameters
    ----------
    points :
        Query point coordinates, shape ``(N, d)``.
    center_point :
        Reference point, shape ``(d,)``.

    Returns
    -------
    :
        Array of distances, shape ``(N,)``.
    """
    return np.linalg.norm(points - center_point, axis=1)


def edges_to_lines(edges: np.ndarray) -> np.ndarray:
    """Convert an edge array to a :mod:`pyvista` lines connectivity array.

    :mod:`pyvista` encodes lines as a flat array where each cell is
    prefixed by its vertex count.  This function prepends ``2`` to each
    edge so the result can be passed directly to
    :class:`pyvista.PolyData`.

    Parameters
    ----------
    edges :
        Edge vertex index pairs, shape ``(E, 2)``.

    Returns
    -------
    :
        Flat connectivity array of length ``3 * E``, alternating between
        the cell size ``2`` and the two vertex indices.
    """
    lines = np.column_stack((np.full((len(edges), 1), 2), edges))
    return lines


def combine_meshes(meshes: list[Mesh]) -> Mesh:
    """Concatenate a list of meshes into a single mesh.

    Vertex arrays are stacked and face index arrays are shifted so that
    each submesh's faces still point to the correct vertices.

    Parameters
    ----------
    meshes :
        List of meshes accepted by :func:`interpret_mesh`.

    Returns
    -------
    vertices :
        Combined vertex array.
    faces :
        Combined face index array with adjusted indices.
    """
    meshes = [interpret_mesh(mesh) for mesh in meshes]
    n_vertices_per_mesh = [mesh[0].shape[0] for mesh in meshes]
    cumulative_n_vertices = list(np.cumsum(n_vertices_per_mesh))
    shifts = [0] + cumulative_n_vertices[:-1]
    vertices = []
    faces = []
    for i, (v, f) in enumerate(meshes):
        vertices.append(v)
        faces.append(f + shifts[i])
    vertices = np.concatenate(vertices, axis=0, dtype=meshes[0][0].dtype)
    faces = np.concatenate(faces, axis=0, dtype=meshes[0][1].dtype)
    return (vertices, faces)


def mesh_connected_components(
    mesh: Mesh,
    size_threshold: Optional[int] = 100,
    sort_by_size: bool = False,
) -> Generator[Mesh, None, None]:
    """Yield each connected component of a mesh as a separate mesh.

    Parameters
    ----------
    mesh :
        Input mesh.
    size_threshold :
        Minimum number of vertices a component must have to be yielded.
        Set to ``None`` to yield all components.
    sort_by_size :
        If ``True``, yield components in descending order of vertex count.

    Yields
    ------
    :
        ``(vertices, faces)`` tuple for each retained component.
    """
    adj = mesh_to_adjacency(mesh)
    _, component_labels = connected_components(adj, directed=False)
    uni_labels, counts = np.unique(component_labels, return_counts=True)
    if size_threshold is not None:
        uni_labels = uni_labels[counts >= size_threshold]
        counts = counts[counts >= size_threshold]

    if sort_by_size:
        sorted_indices = np.argsort(counts)[::-1]
        uni_labels = uni_labels[sorted_indices]

    for label in uni_labels:
        indices = np.where(component_labels == label)[0]
        yield subset_mesh_by_indices(mesh, indices)


def mesh_n_connected_components(mesh: Mesh) -> int:
    """Return the number of connected components in a mesh.

    Parameters
    ----------
    mesh :
        Input mesh.

    Returns
    -------
    :
        Number of connected components.
    """
    adj = mesh_to_adjacency(mesh)
    n_components, _ = connected_components(adj, directed=False)
    return n_components


def threshold_mesh_by_component_size(
    mesh: Mesh, size_threshold: int = 100
) -> tuple[Mesh, np.ndarray]:
    """Remove connected components smaller than a vertex-count threshold.

    Parameters
    ----------
    mesh :
        Input mesh.
    size_threshold :
        Minimum number of vertices a component must have to be retained.

    Returns
    -------
    mesh :
        Filtered ``(vertices, faces)`` tuple containing only vertices and
        faces belonging to components that meet the threshold.
    indices :
        Indices of the retained vertices in the original mesh, shape
        ``(V',)``.
    """
    adj = mesh_to_adjacency(mesh)

    _, component_labels = connected_components(adj, directed=False)
    uni_labels, counts = np.unique(component_labels, return_counts=True)
    uni_labels = uni_labels[counts >= size_threshold]

    mask = np.isin(component_labels, uni_labels)
    indices = np.arange(len(mesh[0]))[mask]
    if len(indices) == 0:
        mesh = (
            np.empty((0, 3), dtype=mesh[0].dtype),
            np.empty((0, 3), dtype=mesh[1].dtype),
        )
        return (mesh, indices)

    mesh = subset_mesh_by_indices(mesh, indices)

    return mesh, indices


def expand_labels(condensed_labels: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Expand per-vertex labels from a reduced mesh back to the original mesh.

    Parameters
    ----------
    condensed_labels :
        Per-vertex label array for the simplified/reduced mesh, shape
        ``(V_simple,)``.
    mapping :
        Array of length ``V_original`` mapping each original vertex to its
        index in the reduced mesh.  Entries of ``-1`` indicate vertices
        that have no corresponding reduced-mesh vertex.

    Returns
    -------
    :
        Per-vertex label array for the original mesh, shape
        ``(V_original,)``.  Vertices with ``mapping == -1`` receive label
        ``-1``.
    """
    labels = condensed_labels[mapping]
    labels[mapping == -1] = -1
    return labels


def graph_to_adjacency(graph: tuple[np.ndarray, np.ndarray]) -> csr_array:
    """Convert a ``(vertices, edges)`` graph tuple to an unweighted adjacency matrix.

    Parameters
    ----------
    graph :
        Tuple of ``(vertices, edges)`` where ``vertices`` has shape
        ``(V, d)`` and ``edges`` has shape ``(E, 2)``.

    Returns
    -------
    :
        Sparse CSR adjacency matrix of shape ``(V, V)`` with entries of
        ``1`` for each edge.
    """
    vertices, edges = graph
    adj = csr_array(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(vertices), len(vertices)),
    )
    return adj


def scale_mesh(mesh: Mesh, scale: float) -> Mesh:
    """Return a new mesh with all vertex positions scaled by ``scale``.

    Parameters
    ----------
    mesh :
        Input mesh accepted by :func:`interpret_mesh`.
    scale :
        Scalar factor to multiply all vertex coordinates by.

    Returns
    -------
    vertices :
        Scaled vertex positions.
    faces :
        Unchanged face index array.
    """
    vertices, faces = interpret_mesh(mesh)
    scaled_vertices = vertices * scale
    return scaled_vertices, faces
