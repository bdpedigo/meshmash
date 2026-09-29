"""Condensing a labeled mesh into a region graph."""

import numpy as np
import pandas as pd
import pytest

from meshmash import compute_vertex_areas
from meshmash.graph import compute_edge_widths, condense_mesh_to_graph
from meshmash.types import interpret_mesh


def _blocky_labels(vertices: np.ndarray, width: float) -> np.ndarray:
    """Regions from a voxel grid, with every 17th vertex unlabeled."""
    cells = np.floor((vertices - vertices.min(axis=0)) / width).astype(np.int64)
    _, labels = np.unique(cells, axis=0, return_inverse=True)
    labels = labels.reshape(-1).astype(np.int64)
    labels[::17] = -1
    return labels


def _condense_over_every_edge(mesh, labels):
    """The same tables, built from one row per mesh edge."""
    faces = mesh[1]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    widths = compute_edge_widths(mesh, mollify_factor=1.0)
    edge_table = pd.DataFrame({"source": edges[:, 0], "target": edges[:, 1]})
    edge_table["boundary_length"] = widths[(edges[:, 0], edges[:, 1])]
    edge_table["count"] = 1
    edge_table["source_group"] = labels[edge_table["source"]]
    edge_table["target_group"] = labels[edge_table["target"]]
    edge_table = edge_table.query(
        "(source_group != -1) and (target_group != -1) and (source_group != target_group)"
    )
    groups = edge_table[["source_group", "target_group"]]
    edge_table = edge_table.assign(
        source_group=groups.min(axis=1), target_group=groups.max(axis=1)
    )
    group_edges = (
        edge_table.groupby(["source_group", "target_group"])
        .agg({"boundary_length": "sum", "count": "sum"})
        .reset_index()
    )

    node_table = pd.DataFrame(mesh[0], columns=["x", "y", "z"])
    node_table["n_vertices"] = np.ones(len(node_table), dtype=np.int32)
    node_table["group"] = labels
    node_table["area"] = compute_vertex_areas(mesh, robust=False)
    agg = {"x": "mean", "y": "mean", "z": "mean", "area": "sum", "n_vertices": "sum"}
    group_nodes = (
        node_table.query("group != -1")
        .groupby("group")
        .agg(agg)
        .loc[np.arange(labels.max() + 1)]
    )
    return group_nodes, group_edges


@pytest.mark.parametrize("add_component_features", [False, True])
def test_matches_the_table_over_every_mesh_edge(mesh, add_component_features):
    """Selecting the crossing edges first changes the cost, not one value."""
    vertices, faces = interpret_mesh(mesh)
    labels = _blocky_labels(vertices, width=2000.0)

    nodes, edges = condense_mesh_to_graph(
        (vertices, faces), labels, add_component_features=add_component_features
    )
    expected_nodes, expected_edges = _condense_over_every_edge(
        (vertices, faces), labels
    )

    assert len(edges) > 0
    np.testing.assert_array_equal(edges["source"], expected_edges["source_group"])
    np.testing.assert_array_equal(edges["target"], expected_edges["target_group"])
    np.testing.assert_array_equal(edges["count"], expected_edges["count"])
    np.testing.assert_array_equal(
        edges["boundary_length"],
        expected_edges["boundary_length"].to_numpy(np.float32),
    )
    for name in ["x", "y", "z", "area", "n_vertices"]:
        np.testing.assert_array_equal(
            nodes[name], expected_nodes[name].to_numpy(nodes[name].dtype)
        )


def test_the_component_columns_follow_the_mesh_components(mesh):
    """Two disjoint copies of a mesh are two components of the same size."""
    vertices, faces = interpret_mesh(mesh)
    shift = np.ptp(vertices, axis=0) * 2
    doubled = (
        np.concatenate([vertices, vertices + shift]),
        np.concatenate([faces, faces + len(vertices)]),
    )
    labels = _blocky_labels(doubled[0], width=2000.0)

    nodes, _ = condense_mesh_to_graph(doubled, labels, add_component_features=True)

    first_copy = labels[: len(vertices)]
    second_copy = labels[len(vertices) :]
    assert set(nodes["component_n_vertices"]) == {
        (first_copy != -1).sum(),
        (second_copy != -1).sum(),
    }
