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
    centroids = group_nodes[["x", "y", "z"]].to_numpy()
    edge_vectors = mesh[0][edge_table["target"]] - mesh[0][edge_table["source"]]
    directions = (
        centroids[edge_table["target_group"]] - centroids[edge_table["source_group"]]
    )
    cosines = np.abs((edge_vectors * directions).sum(axis=1)) / (
        np.linalg.norm(edge_vectors, axis=1) * np.linalg.norm(directions, axis=1)
    )
    edge_table["projected_boundary_length"] = edge_table["boundary_length"] * cosines
    group_edges = (
        edge_table.groupby(["source_group", "target_group"])
        .agg(
            {
                "boundary_length": "sum",
                "projected_boundary_length": "sum",
                "count": "sum",
            }
        )
        .reset_index()
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
    np.testing.assert_allclose(
        edges["projected_boundary_length"],
        expected_edges["projected_boundary_length"],
        rtol=1e-5,
    )
    assert (edges["projected_boundary_length"] <= edges["boundary_length"]).all()
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


def test_a_crossing_edge_sums_the_radii_of_both_its_faces():
    """Six equilateral triangles around a center vertex, center in its own region.

    Every crossing edge is a spoke with two faces, so the boundary is twelve
    incircle radii.
    """
    side = 1000.0
    angles = np.arange(6) * np.pi / 3
    ring = side * np.column_stack([np.cos(angles), np.sin(angles), np.zeros(6)])
    vertices = np.vstack([np.zeros((1, 3)), ring])
    faces = np.array([[0, 1 + i, 1 + (i + 1) % 6] for i in range(6)])
    labels = np.array([0, 1, 1, 1, 1, 1, 1])

    _, edges = condense_mesh_to_graph((vertices, faces), labels)

    # The crossing edges see each side lengthened by a mollify factor of 1.0.
    radius = (side + 1.0) / (2 * np.sqrt(3))
    assert edges["count"].tolist() == [6]
    np.testing.assert_allclose(edges["boundary_length"], 12 * radius, rtol=1e-6)


def test_boundary_length_does_not_depend_on_vertex_order(mesh):
    """Renumbering the vertices moves no edge weight."""
    vertices, faces = interpret_mesh(mesh)
    labels = _blocky_labels(vertices, width=2000.0)
    order = np.random.default_rng(0).permutation(len(vertices))
    new_index = np.empty_like(order)
    new_index[order] = np.arange(len(order))

    _, edges = condense_mesh_to_graph((vertices, faces), labels)
    _, renumbered = condense_mesh_to_graph(
        (vertices[order], new_index[faces]), labels[order]
    )

    assert len(edges) > 0
    np.testing.assert_allclose(
        renumbered["boundary_length"], edges["boundary_length"], rtol=1e-5
    )



def test_coincident_centroids_keep_the_full_boundary_length():
    """Four triangles around a center vertex, center in its own region.

    Both regions are centered on the middle vertex, so there is no direction to
    project onto.
    """
    ring = 1000.0 * np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
    vertices = np.vstack([np.zeros((1, 3)), ring])
    faces = np.array([[0, 1 + i, 1 + (i + 1) % 4] for i in range(4)])
    labels = np.array([0, 1, 1, 1, 1])

    _, edges = condense_mesh_to_graph((vertices, faces), labels)

    np.testing.assert_array_equal(
        edges["projected_boundary_length"], edges["boundary_length"]
    )


def test_a_zero_length_crossing_edge_gives_a_finite_projected_length():
    """Two regions joined across an edge whose two ends are one point."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1000.0, 0.0, 0.0], [1000.0, 0.0, 1.0]]
    )
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    labels = np.array([0, 1, 0, 1])

    _, edges = condense_mesh_to_graph((vertices, faces), labels)

    assert np.isfinite(edges["projected_boundary_length"]).all()
    assert (edges["projected_boundary_length"] <= edges["boundary_length"]).all()
