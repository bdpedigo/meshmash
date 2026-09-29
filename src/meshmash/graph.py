import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from .laplacian import compute_vertex_areas
from .types import Mesh, interpret_mesh
from .utils import connected_components

#: The per-region columns
#: [condense_mesh_to_graph][meshmash.graph.condense_mesh_to_graph] always emits,
#: in the order it emits them, mapped to how each is aggregated over the
#: vertices of a region.  ``x``, ``y`` and ``z`` are the centroid, ``area`` is
#: the summed vertex area, and ``n_vertices`` is the vertex count.
CONDENSED_NODE_PROPERTIES = {
    "x": "mean",
    "y": "mean",
    "z": "mean",
    "area": "sum",
    "n_vertices": "sum",
}

#: The extra columns
#: [condense_mesh_to_graph][meshmash.graph.condense_mesh_to_graph] emits under
#: ``add_component_features``, in order.  Each is a property of the connected
#: component a region sits in, repeated on every region of that component.
CONDENSED_COMPONENT_PROPERTIES = ("component_area", "component_n_vertices")


def condensed_node_property_names(add_component_features: bool = False) -> list[str]:
    """The node-table columns
    [condense_mesh_to_graph][meshmash.graph.condense_mesh_to_graph] emits, in order.

    Derived from
    [CONDENSED_NODE_PROPERTIES][meshmash.graph.CONDENSED_NODE_PROPERTIES] and
    [CONDENSED_COMPONENT_PROPERTIES][meshmash.graph.CONDENSED_COMPONENT_PROPERTIES]
    rather than written out again, so a caller selecting this block by name
    cannot fall out of step with the function that writes it.

    Parameters
    ----------
    add_component_features :
        Whether the caller passed ``add_component_features=True``.

    Returns
    -------
    :
        Column names, of length ``5`` or ``7``.
    """
    names = list(CONDENSED_NODE_PROPERTIES)
    if add_component_features:
        names += list(CONDENSED_COMPONENT_PROPERTIES)
    return names


def _incircle_radii(
    vertices: np.ndarray, faces: np.ndarray, mollify_factor: float
) -> np.ndarray:
    """Incircle radius of each face, from Heron's formula."""
    # ref https://en.wikipedia.org/wiki/Law_of_cotangents

    # let a, b, c be the lengths of the edges of each triangle
    a = (
        np.linalg.norm(vertices[faces[:, 0]] - vertices[faces[:, 1]], axis=1)
        + mollify_factor
    )
    b = (
        np.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 2]], axis=1)
        + mollify_factor
    )
    c = (
        np.linalg.norm(vertices[faces[:, 2]] - vertices[faces[:, 0]], axis=1)
        + mollify_factor
    )
    # s is the semiperimeter of the triangle
    s = (a + b + c) / 2

    return np.sqrt((s - a) * (s - b) * (s - c) / s)


def compute_edge_widths(mesh: Mesh, mollify_factor: float = 0.0) -> csr_array:
    """Compute per-edge width estimates from the incircle radii of adjacent faces.

    For each face the incircle radius is computed from Heron's formula, then
    that radius is summed onto the three edges of the face, so an interior
    edge carries the radii of both its faces.  The
    resulting value at each edge is a geometric proxy for the local "width"
    of the surface at that boundary.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [mesh_to_poly][meshmash.utils.mesh_to_poly].
    mollify_factor :
        Small additive offset applied to each edge length before computing
        face radii.  Prevents division by zero on degenerate faces.

    Returns
    -------
    :
        Symmetric sparse CSR matrix of shape ``(V, V)`` holding, at each edge,
        the summed incircle radii of the faces that share it.
    """
    vertices, faces = mesh
    radii_by_face = _incircle_radii(vertices, faces, mollify_factor)

    r1 = csr_array(
        (radii_by_face, (faces[:, 0], faces[:, 1])),
        shape=(len(vertices), len(vertices)),
    )
    r2 = csr_array(
        (radii_by_face, (faces[:, 1], faces[:, 2])),
        shape=(len(vertices), len(vertices)),
    )
    r3 = csr_array(
        (radii_by_face, (faces[:, 2], faces[:, 0])),
        shape=(len(vertices), len(vertices)),
    )
    radii_adjacency = r1 + r2 + r3

    # Each face writes its edges in its own winding, so the two faces of an
    # edge land on opposite entries; symmetrizing sums both.
    return radii_adjacency + radii_adjacency.T


def _face_components(n_vertices: int, faces: np.ndarray) -> np.ndarray:
    """Connected-component label of each vertex, joined through shared faces."""
    # Two edges per face already join its three corners.
    starts = np.concatenate([faces[:, 0], faces[:, 1]]).astype(np.intc)
    ends = np.concatenate([faces[:, 1], faces[:, 2]]).astype(np.intc)
    graph = csr_array(
        (np.ones(len(starts), dtype=np.int8), (starts, ends)),
        shape=(n_vertices, n_vertices),
    )
    _, component_labels = connected_components(graph, directed=False)
    return component_labels


def _crossing_edges(
    vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each mesh edge between two labeled regions, once, with its width.

    Returns the lower and upper vertex of each edge, sorted by that pair, and
    the edge's width.
    """
    # Only these edges reach the edge table, so they are selected per face
    # edge before any array over every mesh edge exists. That array was the
    # memory peak on a large mesh.
    radii = _incircle_radii(vertices, faces, mollify_factor=1.0)
    crossing_starts, crossing_ends, crossing_widths = [], [], []
    for start_corner, end_corner in ((0, 1), (1, 2), (2, 0)):
        starts = faces[:, start_corner]
        ends = faces[:, end_corner]
        start_labels = labels[starts]
        end_labels = labels[ends]
        crossing = (
            (start_labels != -1) & (end_labels != -1) & (start_labels != end_labels)
        )
        crossing_widths.append(radii[crossing])
        crossing_starts.append(starts[crossing])
        crossing_ends.append(ends[crossing])
    # int64 before the key is formed: faces are often uint32, and numpy 1.x
    # keeps uint32 * scalar in uint32, where V * V overflows.
    starts = np.concatenate(crossing_starts).astype(np.int64)
    ends = np.concatenate(crossing_ends).astype(np.int64)
    n_vertices = len(vertices)
    edge_keys = np.minimum(starts, ends) * n_vertices + np.maximum(starts, ends)
    edge_keys, edge_index = np.unique(edge_keys, return_inverse=True)
    widths = np.bincount(
        edge_index, weights=np.concatenate(crossing_widths), minlength=len(edge_keys)
    ).astype(radii.dtype)
    return edge_keys // n_vertices, edge_keys % n_vertices, widths


def condense_mesh_to_graph(
    mesh: Mesh, labels: np.ndarray, add_component_features: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Condense a vertex-labeled mesh into a node-edge graph representation.

    Each unique label becomes a node; adjacent regions (labels that share at
    least one mesh edge) become connected by an edge.  Edge weights reflect
    the total boundary length between regions.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [mesh_to_poly][meshmash.utils.mesh_to_poly].
    labels :
        Per-vertex integer label array of length ``V``.  Vertices with
        label ``-1`` are excluded.
    add_component_features :
        If ``True``, add a ``component`` column to the node table indicating
        which connected component (in the *original* mesh graph) each
        label region belongs to.

    Returns
    -------
    node_table :
        DataFrame indexed by an ``int32`` label, with the columns
        [condensed_node_property_names][meshmash.graph.condensed_node_property_names]
        gives for the same ``add_component_features``.  Vertex counts are
        ``int32`` and every other column is ``float32``.
    edge_table :
        DataFrame with ``int32`` columns ``source``, ``target`` and ``count``
        (number of mesh edges crossing the boundary), and ``float32`` columns
        ``boundary_length`` (sum of edge-width values) and ``edge_length``
        (distance between the two centroids).
    """
    vertices, faces = interpret_mesh(mesh)
    labels = np.asarray(labels)

    lower, upper, boundary_lengths = _crossing_edges(vertices, faces, labels)
    source_labels = labels[lower]
    target_labels = labels[upper]

    group_edge_table = (
        pd.DataFrame(
            {
                "source_group": np.minimum(source_labels, target_labels),
                "target_group": np.maximum(source_labels, target_labels),
                "boundary_length": boundary_lengths,
                "count": np.ones(len(lower), dtype=np.int64),
            }
        )
        .groupby(["source_group", "target_group"])
        .agg({"boundary_length": "sum", "count": "sum"})
        .reset_index()
    )

    areas = compute_vertex_areas(mesh, robust=False)

    labeled = labels != -1
    node_table = pd.DataFrame(vertices[labeled], columns=["x", "y", "z"])
    node_table["n_vertices"] = np.ones(len(node_table), dtype=np.int32)
    node_table["group"] = labels[labeled]
    node_table["area"] = areas[labeled]

    agg_dict = dict(CONDENSED_NODE_PROPERTIES)

    if add_component_features:
        node_table["component"] = _face_components(len(vertices), faces)[labeled]
        agg_dict["component"] = "first"

    group_node_table = (
        node_table.groupby(["group"])
        .agg(agg_dict)
        .loc[np.arange(labels.max() + 1)]  # make sure we are indexed correctly
    )

    if add_component_features:
        component_area = group_node_table.groupby("component")["area"].sum()
        group_node_table["component_area"] = group_node_table["component"].map(
            component_area
        )
        component_n_vertices = group_node_table.groupby("component")["n_vertices"].sum()
        group_node_table["component_n_vertices"] = group_node_table["component"].map(
            component_n_vertices
        )
        group_node_table.drop("component", axis=1, inplace=True)

    group_edge_table["edge_length"] = np.linalg.norm(
        group_node_table.loc[group_edge_table["source_group"]][["x", "y", "z"]].values
        - group_node_table.loc[group_edge_table["target_group"]][
            ["x", "y", "z"]
        ].values,
        axis=1,
    )

    group_edge_table.rename(
        {"source_group": "source", "target_group": "target"}, axis=1, inplace=True
    )
    group_node_table.index.name = None

    # Summed and differenced in float64 above; stored at the precision the
    # features beside them carry.
    group_node_table = group_node_table.astype(
        {
            name: np.int32 if name.endswith("n_vertices") else np.float32
            for name in group_node_table.columns
        }
    )
    group_node_table.index = group_node_table.index.astype(np.int32)
    group_edge_table = group_edge_table.astype(
        {
            "source": np.int32,
            "target": np.int32,
            "boundary_length": np.float32,
            "count": np.int32,
            "edge_length": np.float32,
        }
    )

    return group_node_table, group_edge_table
