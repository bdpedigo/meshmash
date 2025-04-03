import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from .laplacian import compute_vertex_areas
from .utils import connected_components, mesh_to_adjacency, mesh_to_edges


def compute_edge_widths(mesh, mollify_factor: float = 0.0) -> csr_array:
    # ref https://en.wikipedia.org/wiki/Law_of_cotangents

    vertices, faces = mesh
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

    radii_by_face = np.sqrt((s - a) * (s - b) * (s - c) / s)

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

    return radii_adjacency


def condense_mesh_to_graph(
    mesh, labels, add_component_features: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = mesh_to_edges(mesh)

    sources, targets = edges[:, 0], edges[:, 1]

    radii_adjacency = compute_edge_widths(mesh, mollify_factor=1.0)

    edge_table = pd.DataFrame(
        {
            "source": sources,
            "target": targets,
        }
    )
    edge_table["boundary_length"] = radii_adjacency[(sources, targets)]
    edge_table["count"] = 1

    edge_table["source_group"] = labels[edge_table["source"]]
    edge_table["target_group"] = labels[edge_table["target"]]

    edge_table.query(
        "(source_group != -1) and (target_group != -1) and (source_group != target_group)",
        inplace=True,
    )

    group_edge_table = (
        edge_table.groupby(["source_group", "target_group"])
        .agg({"boundary_length": "sum", "count": "sum"})
        .reset_index()
    )

    areas = compute_vertex_areas(mesh, robust=False)

    node_table = pd.DataFrame(mesh[0], columns=["x", "y", "z"])
    node_table["n_vertices"] = 1
    node_table["group"] = labels
    node_table["area"] = areas

    agg_dict = {
        "x": "mean",
        "y": "mean",
        "z": "mean",
        "area": "sum",
        "n_vertices": "sum",
    }

    if add_component_features:
        adj = mesh_to_adjacency(mesh)

        _, cc_labels = connected_components(adj, directed=False)
        node_table["component"] = cc_labels
        agg_dict["component"] = "first"

    node_table.query("group != -1", inplace=True)

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

    return group_node_table, group_edge_table
