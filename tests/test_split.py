"""Tests for the splitting methods and the queue they share.

Every mesh here is synthetic, so the tests need no download.  They also avoid
the eigensolver: the recursive spectral bisection is not reproducible run to run (a
vertex inside the solver's tolerance lands on either side), so the only way to
pin its loop is to feed it a cut that is.
"""

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from meshmash import (
    MeshStitcher,
    fit_mesh_split_geodesic,
    fit_mesh_split_spectral,
    mesh_to_adjacency,
)
from meshmash.split import (
    _fit_split_by_queue,
    spectral_bisect,
    spectral_bisect_adjacency,
)


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
    a chunk cannot come back in two pieces.  The spectral cut makes no such
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
    """A second component under the threshold is dropped, as in the spectral cut."""
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


def face_less_mapping(tube):
    """A partition whose middle chunk owns one vertex, and so owns no face.

    A face joins a chunk only when all three of its vertices carry the label,
    so a single vertex, and equally a cell one vertex wide, contributes none.
    The geodesic cut can leave such a cell; here it is built by hand so the
    test does not depend on finding a mesh that provokes one.
    """
    n_vertices = len(tube[0])
    submesh_mapping = np.zeros(n_vertices, dtype=int)
    submesh_mapping[n_vertices // 2 :] = 2
    submesh_mapping[n_vertices // 4] = 1
    return submesh_mapping


def test_expand_split_keeps_a_chunk_that_owns_no_face(tube):
    """Chunk count comes from the partition, not from the faces that survive it.

    `apply_mesh_split` drops a label with no face of its own.  Taking the
    loop bounds from that list drops the trailing chunks and pairs the rest
    with the wrong core.
    """
    submesh_mapping = face_less_mapping(tube)
    stitcher = MeshStitcher(tube, n_jobs=1)

    submeshes = stitcher.expand_split(submesh_mapping, overlap_distance=5.0)

    assert len(submeshes) == 3
    for label in (0, 1, 2):
        core = np.nonzero(submesh_mapping == label)[0]
        overlap = stitcher.submesh_overlap_indices[label]
        assert np.isin(core, overlap).all(), (
            f"chunk {label} must contain the vertices it owns"
        )


def test_stitching_covers_every_vertex_of_a_face_less_partition(tube):
    """The symptom downstream: a lost chunk leaves its vertices unfilled."""
    submesh_mapping = face_less_mapping(tube)
    stitcher = MeshStitcher(tube, n_jobs=1)
    stitcher.expand_split(submesh_mapping, overlap_distance=5.0)

    stitched = stitcher.apply(lambda submesh: submesh[0][:, 2:3])

    assert np.allclose(stitched[:, 0], tube[0][:, 2])


def test_expand_split_rejects_labels_with_a_gap(tube):
    """Position and label are the same number, so a gap has no right answer."""
    submesh_mapping = np.zeros(len(tube[0]), dtype=int)
    submesh_mapping[len(tube[0]) // 2 :] = 2
    stitcher = MeshStitcher(tube, n_jobs=1)

    with pytest.raises(ValueError):
        stitcher.expand_split(submesh_mapping, overlap_distance=5.0)


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


def test_spectral_split_takes_an_adjacency_matrix(tube):
    """The polymorphic entry `fit_mesh_split_spectral` documents, kept by the
    refactor."""
    adjacency = mesh_to_adjacency(tube)

    def halve(adj):
        half = adj.shape[0] // 2
        one = np.arange(half)
        two = np.arange(half, adj.shape[0])
        return (adj[one][:, one], adj[two][:, two]), (one, two)

    from meshmash import split as split_module

    original = split_module.spectral_bisect_adjacency
    split_module.spectral_bisect_adjacency = halve
    try:
        labels = fit_mesh_split_spectral(adjacency, max_vertex_threshold=500)
    finally:
        split_module.spectral_bisect_adjacency = original

    assert len(labels) == adjacency.shape[0]
    assert chunk_sizes(labels).max() <= 500


def reference_fit_mesh_split_spectral(
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

    The cut here is a deterministic index halving, because the real spectral
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
        halve,
        max_vertex_threshold=300,
        min_vertex_threshold=100,
        max_rounds=100_000,
    )
    expected = reference_fit_mesh_split_spectral(
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
        spectral_bisect_adjacency,
        max_vertex_threshold=100,
        min_vertex_threshold=100,
        max_rounds=0,
    )

    assert (labels == -1).all()


def test_a_seed_makes_the_partition_reproducible(tube):
    """Without one, ARPACK draws its own start vector and the cut moves.

    The variation is not in the method. ARPACK's generator carries state across
    calls inside a process, so the second bisection in a session starts
    somewhere else and lands on a slightly different cut. Everything
    downstream of the partition moves with it.
    """
    runs = []
    for _ in range(3):
        stitcher = MeshStitcher(tube, n_jobs=1)
        stitcher.split_mesh(
            max_vertex_threshold=500,
            overlap_distance=5.0,
            verify_connected=False,
            seed=7,
        )
        runs.append(stitcher.submesh_mapping.copy())

    assert runs[0].max() > 0, "one chunk would prove nothing"
    for other in runs[1:]:
        np.testing.assert_array_equal(runs[0], other)


def test_a_retry_draws_a_different_vector(tube):
    """The retry has to move, or it repeats the cut that just failed.

    `spectral_bisect_adjacency` retries when a cut leaves a vertex isolated.
    Under one fixed seed every attempt would redraw the identical vector and
    return the identical bad cut until the attempts ran out, so the seed is
    offset by the attempt number.
    """
    adjacency = mesh_to_adjacency(tube)
    seen = [spectral_bisect(adjacency, seed=5 + attempt)[0] for attempt in range(3)]

    assert not all(np.array_equal(seen[0], other) for other in seen[1:]), (
        "consecutive attempt seeds must not give the same cut"
    )


def test_the_sign_anchor_survives_a_vertex_zero_at_the_cut(tube):
    """The cut is a partition, and must not depend on which vertex is first.

    The Fiedler vector's sign is arbitrary, and fixing it decides which side is
    called "1". Anchoring that on vertex 0 reads a number that can sit
    anywhere, including at the cut where its sign is noise; a vertex 0 of
    exactly zero would have zeroed the whole vector and sent every vertex to
    one side. Reordering the graph must leave the partition alone.
    """
    adjacency = mesh_to_adjacency(tube)
    n = adjacency.shape[0]
    first, second = spectral_bisect(adjacency, seed=0)

    # Move a vertex that sits near the cut into position 0.
    order = np.argsort(np.isin(np.arange(n), first[:1]).astype(int))[::-1].copy()
    reordered = adjacency[order][:, order]
    moved_first, moved_second = spectral_bisect(reordered, seed=0)

    sizes = sorted([len(first), len(second)])
    moved_sizes = sorted([len(moved_first), len(moved_second)])
    assert min(sizes) > 0 and min(moved_sizes) > 0
    assert sizes == moved_sizes
