import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from .laplacian import compute_vertex_areas
from .utils import mesh_to_edges


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


def condense_mesh_to_graph(mesh, labels):
    edges = mesh_to_edges(mesh)

    sources, targets = edges[:, 0], edges[:, 1]

    radii_adjacency = compute_edge_widths(mesh, mollify_factor=1.0)

    edge_table = pd.DataFrame(
        {
            "source": sources,
            "target": targets,
        }
    )
    edge_table["width"] = radii_adjacency[(sources, targets)]
    edge_table["count"] = 1

    edge_table["source_group"] = labels[edge_table["source"]]
    edge_table["target_group"] = labels[edge_table["target"]]

    edge_table.query(
        "(source_group != -1) and (target_group != -1) and (source_group != target_group)",
        inplace=True,
    )

    edge_table["length"] = np.linalg.norm(
        mesh[0][edge_table["source"].values] - mesh[0][edge_table["target"].values],
        axis=1,
    )

    group_edge_table = (
        edge_table.groupby(["source_group", "target_group"])
        .agg({"width": "sum", "count": "sum"})
        .reset_index()
    )

    areas = compute_vertex_areas(mesh, robust=False)

    node_table = pd.DataFrame(mesh[0], columns=["x", "y", "z"])
    node_table["count"] = 1
    node_table["group"] = labels
    node_table["area"] = areas
    node_table.query("group != -1", inplace=True)

    group_node_table = (
        node_table.groupby(["group"])
        .agg({"x": "mean", "y": "mean", "z": "mean", "area": "sum", "count": "sum"})
        .loc[np.arange(labels.max() + 1)]  # make sure we are indexed correctly
    )

    group_edge_table["length"] = np.linalg.norm(
        group_node_table.loc[group_edge_table["source_group"]][["x", "y", "z"]].values
        - group_node_table.loc[group_edge_table["target_group"]][
            ["x", "y", "z"]
        ].values,
        axis=1,
    )

    return group_node_table, group_edge_table
