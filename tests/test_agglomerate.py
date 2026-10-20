"""Cutting a chunk into domains: each owned component alone, means over the domain.

The meshes here are flat grids built in the test, so nothing needs a download
or an eigensolver. The features are smooth and positive, so Ward on their log
merges a whole connected piece into one domain at the default threshold.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from meshmash import agglomerate_components, compute_vertex_areas, condense_features


def grid_mesh(n_x=30, n_y=10):
    """A flat ``n_x`` by ``n_y`` grid of unit squares, two triangles each."""
    x, y = np.meshgrid(np.arange(n_x), np.arange(n_y), indexing="ij")
    vertices = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1) * 100.0
    index = np.arange(n_x * n_y).reshape(n_x, n_y)
    corner = index[:-1, :-1].ravel()
    right = index[1:, :-1].ravel()
    up = index[:-1, 1:].ravel()
    diagonal = index[1:, 1:].ravel()
    faces = np.concatenate(
        [np.stack([corner, right, diagonal], 1), np.stack([corner, diagonal, up], 1)]
    )
    return vertices.astype(np.float64), faces.astype(np.int64)


def smooth_features(vertices, n_columns=3):
    """Positive and slowly varying, so one connected piece is one domain."""
    return 1.0 + 1e-4 * vertices[:, [0]] + np.zeros((1, n_columns))


def disconnected_labels(labels, faces):
    """The labels whose vertices are not connected through same-label edges."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    same = labels[edges[:, 0]] == labels[edges[:, 1]]
    edges = edges[same & (labels[edges[:, 0]] >= 0)]
    graph = sparse.coo_array(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(labels), len(labels)),
    )
    _, piece = connected_components(graph, directed=False)
    assigned = labels >= 0
    pieces = pd.DataFrame({"label": labels[assigned], "piece": piece[assigned]})
    n_pieces = pieces.groupby("label")["piece"].nunique()
    return n_pieces.index[n_pieces > 1].tolist()


@pytest.fixture
def grid():
    return grid_mesh()


@pytest.fixture
def stripes(grid):
    """Left and right stripes owned by one chunk, the middle stripe its band."""
    x = grid[0][:, 0] / 100.0
    return (x < 10) | (x >= 20), (x >= 10) & (x < 20)


def test_a_chunk_boundary_that_cuts_a_cluster_leaves_every_domain_connected(
    grid, stripes
):
    """The failure the owned cut exists for, and its fix.

    Clustered over the whole patch, the two owned stripes join through the band
    into one cluster, and the owned part of that cluster is in two pieces.
    Clustered on the owned vertices, each stripe is a domain of its own.
    """
    owned, band = stripes
    features = smooth_features(grid[0])

    _, patch_labels = condense_features(grid, features)
    patch_labels = np.where(owned, patch_labels[:, 0], -1)
    assert disconnected_labels(patch_labels, grid[1]), "the premise of the test"

    _, labels = condense_features(grid, features, owned=owned)
    labels = labels[:, 0]

    assert disconnected_labels(labels, grid[1]) == []
    assert (labels[band] == -1).all()
    assert (labels[owned] >= 0).all()
    assert len(np.unique(labels[owned])) == 2


def test_non_finite_vertices_are_left_out_before_the_components(grid, stripes):
    """A band of non-finite features splits a piece in two, and Ward sees two."""
    owned, band = stripes
    log_features = np.log(smooth_features(grid[0]))
    log_features[band] = -np.inf

    labels = agglomerate_components(grid, log_features, 3.0)

    assert labels.dtype == np.int32
    assert (labels[band] == -1).all()
    assert disconnected_labels(labels, grid[1]) == []
    assert len(np.unique(labels[owned])) == 2


def test_a_vertex_in_no_face_is_a_domain_of_its_own(grid):
    vertices, faces = grid
    vertices = np.concatenate([vertices, [[1e5, 1e5, 0.0]]])
    log_features = np.log(smooth_features(vertices))

    labels = agglomerate_components((vertices, faces), log_features, 3.0)

    assert labels[-1] >= 0
    assert (labels[:-1] != labels[-1]).all()


def test_the_means_are_over_the_domains_own_vertices(grid, stripes):
    """A band value no owned vertex holds must not reach any domain's mean."""
    owned, band = stripes
    features = pd.DataFrame(smooth_features(grid[0]), columns=["a", "b", "c"])
    aggregated = features.copy()
    aggregated["c"] = np.where(band, 1e6, 1.0)

    means, labels = condense_features(
        grid, aggregated, cluster_features=features, owned=owned
    )
    labels = labels[:, 0]

    areas = compute_vertex_areas(grid)
    for label in np.unique(labels[labels >= 0]):
        members = labels == label
        expected = np.average(aggregated.loc[members, "a"], weights=areas[members])
        np.testing.assert_allclose(means.loc[label, "a"], expected, rtol=1e-6)
    assert (means.drop(index=-1)["c"] == 1.0).all()


def test_owned_must_be_per_vertex(grid):
    features = smooth_features(grid[0])
    with pytest.raises(ValueError, match="per-vertex"):
        condense_features(grid, features, owned=np.ones(len(features) - 1, bool))
