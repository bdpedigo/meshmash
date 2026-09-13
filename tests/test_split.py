"""Tests for the splitting methods and the queue they share.

Every mesh here is synthetic, so the tests need no download.  They also avoid
the eigensolver: the recursive bisection is not reproducible run to run (a
vertex inside the solver's tolerance lands on either side), so the only way to
pin its loop is to feed it a cut that is.
"""

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from meshmash import (
    MeshStitcher,
    fit_mesh_split,
    fit_mesh_split_geodesic,
    fit_mesh_split_lap,
    mesh_to_adjacency,
)
from meshmash.split import _fit_split_by_queue, bisect_adjacency


def tube_mesh(n_rings=100, n_theta=40, radius=5.0, spacing=1.0):
    """A closed tube of ``n_rings * n_theta`` vertices, built here so the tests
    need no download."""
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z = np.arange(n_rings) * spacing
    vertices = np.stack(
        [
            np.tile(radius * np.cos(theta), n_rings),
            np.tile(radius * np.sin(theta), n_rings),
            np.repeat(z, n_theta),
        ],
        axis=1,
    ).astype(np.float64)

    faces = []
    for ring in range(n_rings - 1):
        for step in range(n_theta):
            here = ring * n_theta + step
            right = ring * n_theta + (step + 1) % n_theta
            faces.append([here, right, here + n_theta])
            faces.append([right, right + n_theta, here + n_theta])
    return vertices, np.array(faces, dtype=np.int64)


@pytest.fixture(scope="module")
def tube():
    """A tube of 4,000 vertices, big enough to need several chunks."""
    return tube_mesh()


def chunk_sizes(labels):
    return np.bincount(labels[labels != -1])


def test_geodesic_labels_meet_the_split_contract(tube):
    labels = fit_mesh_split_geodesic(tube, max_vertex_threshold=500)

    assert len(labels) == len(tube[0])
    assert labels.min() == 0, "one connected tube leaves no vertex unassigned"
    present = np.unique(labels)
    assert np.array_equal(present, np.arange(present.max() + 1)), (
        "labels must be contiguous, since the partition indexes a chunk list"
    )
    sizes = chunk_sizes(labels)
    assert sizes.max() <= 500
    assert np.array_equal(sizes, np.sort(sizes)[::-1]), "largest chunk first"


def test_geodesic_chunks_are_connected(tube):
    """The property the method rests on: a cell of a connected piece is connected.

    Any vertex on a shortest path to its owning seed is owned by that seed, so
    a chunk cannot come back in two pieces.  The bisection makes no such
    promise, which is why nothing downstream can assume it.
    """
    labels = fit_mesh_split_geodesic(tube, max_vertex_threshold=500)
    adjacency = mesh_to_adjacency(tube)

    for label in np.unique(labels):
        indices = np.nonzero(labels == label)[0]
        sub = adjacency[indices][:, indices]
        assert connected_components(sub, directed=False)[0] == 1


def test_geodesic_is_reproducible(tube):
    first = fit_mesh_split_geodesic(tube, max_vertex_threshold=500)
    second = fit_mesh_split_geodesic(tube, max_vertex_threshold=500)

    assert np.array_equal(first, second)


def test_geodesic_target_vertices_sets_the_chunk_size(tube):
    coarse = fit_mesh_split_geodesic(
        tube, max_vertex_threshold=2_000, target_vertices=1_000
    )
    fine = fit_mesh_split_geodesic(
        tube, max_vertex_threshold=2_000, target_vertices=250
    )

    assert len(chunk_sizes(fine)) > len(chunk_sizes(coarse))


@pytest.mark.parametrize("target_vertices", [0, -1, -1_000])
def test_geodesic_rejects_a_non_positive_target(tube, target_vertices):
    """The seed count divides by this, so a bad value must fail at the door.

    Zero divides by zero inside the cut, and a negative value fails
    silently: ``max(2, ...)`` turns it into a two-cell cut that ignores the
    caller.  Neither is an answer to give back.
    """
    with pytest.raises(ValueError):
        fit_mesh_split_geodesic(
            tube, max_vertex_threshold=500, target_vertices=target_vertices
        )


def test_geodesic_drops_small_components(tube):
    """A second component under the threshold is dropped, as in the bisection."""
    vertices, faces = tube
    small_vertices, small_faces = tube_mesh(n_rings=3, n_theta=10)
    small_vertices = small_vertices + np.array([0.0, 0.0, 1_000.0])
    both = (
        np.concatenate([vertices, small_vertices]),
        np.concatenate([faces, small_faces + len(vertices)]),
    )

    labels = fit_mesh_split_geodesic(
        both, max_vertex_threshold=500, min_vertex_threshold=100
    )

    assert (labels[len(vertices) :] == -1).all()
    assert (labels[: len(vertices)] != -1).all()


def test_stitcher_runs_the_whole_workflow_on_geodesic_chunks(tube):
    """Split, compute per chunk, stitch back, with only the cut swapped."""
    stitcher = MeshStitcher(tube, n_jobs=1)
    submeshes = stitcher.split_mesh(
        max_vertex_threshold=500,
        target_vertices=400,
        method="geodesic",
        overlap_distance=5.0,
    )

    assert len(submeshes) == len(np.unique(stitcher.submesh_mapping))

    # Each chunk reports its own vertex heights; stitching must put every
    # vertex back where it came from.
    stitched = stitcher.apply(lambda submesh: submesh[0][:, 2:3])

    assert np.allclose(stitched[:, 0], tube[0][:, 2])


def test_split_mesh_rejects_an_unknown_method(tube):
    stitcher = MeshStitcher(tube, n_jobs=1)

    with pytest.raises(ValueError):
        stitcher.split_mesh(method="metis")


def test_lap_split_refuses_an_adjacency_matrix(tube):
    """It never worked: the cotangent Laplacian needs vertex positions."""
    with pytest.raises(TypeError):
        fit_mesh_split_lap(mesh_to_adjacency(tube))


def test_bisection_takes_an_adjacency_matrix(tube):
    """The polymorphic entry `fit_mesh_split` documents, kept by the refactor."""
    adjacency = mesh_to_adjacency(tube)

    def halve(adj):
        half = adj.shape[0] // 2
        one = np.arange(half)
        two = np.arange(half, adj.shape[0])
        return (adj[one][:, one], adj[two][:, two]), (one, two)

    from meshmash import split as split_module

    original = split_module.bisect_adjacency
    split_module.bisect_adjacency = halve
    try:
        labels = fit_mesh_split(adjacency, max_vertex_threshold=500)
    finally:
        split_module.bisect_adjacency = original

    assert len(labels) == adjacency.shape[0]
    assert chunk_sizes(labels).max() <= 500


def reference_fit_mesh_split(
    whole_adj, cut, max_vertex_threshold, min_vertex_threshold, max_rounds
):
    """The loop as it stood before the queue was shared, for comparison only."""
    n_vertices = whole_adj.shape[0]
    mesh_indices = np.arange(n_vertices)
    n_components, component_labels = connected_components(whole_adj)

    adj_queue = []
    for component_id in range(n_components):
        component_mask = component_labels == component_id
        if component_mask.sum() >= min_vertex_threshold:
            component_indices = mesh_indices[component_mask]
            component_adj = whole_adj[component_indices][:, component_indices]
            adj_queue.append((component_adj, component_indices))

    submesh_mapping = np.full(n_vertices, -1, dtype=int)
    n_finished = 0
    rounds = 0
    while len(adj_queue) > 0 and rounds < max_rounds:
        current_adj, current_indices = adj_queue.pop(0)
        if current_adj.shape[0] <= max_vertex_threshold:
            sub_adjs, submesh_indices_to_main = [current_adj], [current_indices]
        else:
            sub_adjs, submesh_indices = cut(current_adj)
            submesh_indices_to_main = [
                current_indices[indices] for indices in submesh_indices
            ]
        for sub_adj, indices in zip(sub_adjs, submesh_indices_to_main):
            if sub_adj.shape[0] > max_vertex_threshold:
                adj_queue.append((sub_adj, indices))
            else:
                submesh_mapping[indices] = n_finished
                n_finished += 1
        rounds += 1

    valid = submesh_mapping[submesh_mapping != -1]
    labels, counts = np.unique(valid, return_counts=True)
    reorder = np.argsort(-counts)
    new_labels = np.arange(labels.max() + 1)
    old_to_new = dict(zip(labels[reorder], new_labels))
    old_to_new[-1] = -1
    return np.vectorize(old_to_new.get)(submesh_mapping)


def test_shared_queue_reproduces_the_loop_it_replaced(tube):
    """Same cut, same partition as the per-method loop that came before.

    The cut here is a deterministic index halving, because the real Fiedler
    cut does not repeat itself and cannot settle the question.
    """
    adjacency = mesh_to_adjacency(tube)

    def halve(adj):
        half = adj.shape[0] // 2
        one = np.arange(half)
        two = np.arange(half, adj.shape[0])
        return (adj[one][:, one], adj[two][:, two]), (one, two)

    shared = _fit_split_by_queue(
        adjacency,
        lambda indices: adjacency[indices][:, indices],
        lambda adj: adj.shape[0],
        halve,
        max_vertex_threshold=300,
        min_vertex_threshold=100,
        max_rounds=100_000,
    )
    expected = reference_fit_mesh_split(
        adjacency,
        halve,
        max_vertex_threshold=300,
        min_vertex_threshold=100,
        max_rounds=100_000,
    )

    assert np.array_equal(shared, expected)


def test_max_rounds_stops_the_queue(tube):
    """The runaway guard leaves the unfinished vertices in no chunk."""
    adjacency = mesh_to_adjacency(tube)

    labels = _fit_split_by_queue(
        adjacency,
        lambda indices: adjacency[indices][:, indices],
        lambda adj: adj.shape[0],
        bisect_adjacency,
        max_vertex_threshold=100,
        min_vertex_threshold=100,
        max_rounds=0,
    )

    assert (labels == -1).all()
