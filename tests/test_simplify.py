import subprocess
import sys

import numpy as np
import pytest
from fast_simplification import replay_simplification, simplify
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from meshmash import simplify_mesh, simplify_to_density
from meshmash.simplify import _decimate
from meshmash.utils import remove_repeated_vertex_faces, vertex_density


@pytest.fixture(scope="module")
def simplified(mesh):
    return simplify_mesh(mesh, target_reduction=0.7)


def longest_edge(vertices, faces):
    corners = vertices[faces]
    return max(
        np.linalg.norm(corners[:, a] - corners[:, b], axis=1).max()
        for a, b in [(0, 1), (1, 2), (2, 0)]
    )


@pytest.fixture(scope="module")
def mesh_with_repeated_vertex_faces(mesh):
    """The sample mesh plus faces like ``(a, a, new)``, as real meshes have.

    Each ``new`` vertex sits in the middle of the vertex array and no other
    face uses it, so every index after it moves if a decimation keeps it in
    one array and drops it from another.
    """
    vertices, faces = mesh
    rng = np.random.default_rng(0)
    middle = len(vertices) // 2
    n_new = 20
    anchors = rng.choice(len(vertices), size=n_new, replace=False)
    new_positions = vertices[anchors] + 10.0
    vertices = np.insert(vertices, middle, new_positions, axis=0)
    faces = np.where(faces >= middle, faces + n_new, faces)
    anchors = np.where(anchors >= middle, anchors + n_new, anchors)
    new_ids = middle + np.arange(n_new)
    extra = np.stack([anchors, anchors, new_ids], axis=1)
    faces = np.concatenate([faces, extra]).astype(mesh[1].dtype)
    return vertices, faces


def test_simplify_to_density_keeps_edges_local(mesh_with_repeated_vertex_faces):
    """No simplified edge spans the mesh.

    A mapping off by even one vertex sends faces across the whole mesh: on a
    real neuron it gave edges of 756 um against an input maximum of 1 um.
    """
    vertices, faces = mesh_with_repeated_vertex_faces
    target = vertex_density((vertices, faces)) * 0.3
    new_vertices, new_faces, _ = simplify_to_density(
        (vertices, faces), target_density=target
    )
    assert longest_edge(new_vertices, new_faces) < 20 * longest_edge(vertices, faces)


def test_simplify_mesh_reduces_the_face_count(mesh, simplified):
    (_, faces), _ = simplified
    assert faces.shape[0] < mesh[1].shape[0]


def test_simplify_mesh_mapping_covers_every_input_vertex(mesh, simplified):
    _, mapping = simplified
    assert mapping.shape == (mesh[0].shape[0],)


def test_simplify_mesh_mapping_indexes_the_simplified_mesh(simplified):
    (vertices, _), mapping = simplified
    assert mapping.min() >= 0
    assert mapping.max() < vertices.shape[0]


def test_simplify_mesh_mapping_agrees_with_the_returned_faces(simplified):
    """Remapping the input's own faces lands on real vertices of the mesh that came back."""
    (vertices, faces), mapping = simplified
    assert faces.max() < vertices.shape[0]
    assert set(np.unique(faces)).issubset(set(np.unique(mapping)))


def test_simplify_mesh_none_is_a_pass_through(mesh):
    (vertices, faces), mapping = simplify_mesh(mesh, target_reduction=None)
    np.testing.assert_array_equal(vertices, mesh[0])
    np.testing.assert_array_equal(faces, mesh[1])
    np.testing.assert_array_equal(mapping, np.arange(mesh[0].shape[0]))


def test_simplify_mesh_reduction_controls_how_much_is_removed(mesh):
    (_, light), _ = simplify_mesh(mesh, target_reduction=0.2)
    (_, heavy), _ = simplify_mesh(mesh, target_reduction=0.9)
    assert heavy.shape[0] < light.shape[0] < mesh[1].shape[0]


def unreferenced(vertices, faces):
    """The one vertex no face uses."""
    (index,) = np.setdiff1d(np.arange(len(vertices)), faces)
    return index


@pytest.fixture(scope="module")
def mesh_with_unreferenced_vertex(mesh):
    """The sample mesh plus a vertex no face uses, in the middle of the array."""
    vertices, faces = mesh
    middle = len(vertices) // 2
    vertices = np.insert(vertices, middle, vertices[0] + 1e6, axis=0)
    faces = np.where(faces >= middle, faces + 1, faces).astype(faces.dtype)
    return vertices, faces


@pytest.fixture(scope="module")
def mesh_with_collapsing_strip(mesh):
    """The sample mesh plus a separate 10 nm wide strip that decimates to a line."""
    vertices, faces = mesh
    n = 10
    xs = np.arange(n) * 200.0
    rails = [np.stack([xs, np.full(n, y), np.zeros(n)], axis=1) for y in (0.0, 10.0)]
    strip = np.concatenate(rails) + vertices.max(axis=0) + 5e3
    i = np.arange(n - 1)
    strip_faces = len(vertices) + np.concatenate(
        [
            np.stack([i, i + 1, i + n], axis=1),
            np.stack([i + 1, i + n + 1, i + n], axis=1),
        ]
    )
    return (
        np.concatenate([vertices, strip.astype(vertices.dtype)]),
        np.concatenate([faces, strip_faces]).astype(faces.dtype),
    )


DECIMATION_CASES = [
    ("mesh", 0.3),
    ("mesh", 0.7),
    ("mesh", 0.9),
    ("mesh_with_repeated_vertex_faces", 0.7),
    ("mesh_with_unreferenced_vertex", 0.7),
    ("mesh_with_collapsing_strip", 0.7),
]


@pytest.fixture(params=DECIMATION_CASES, ids=lambda case: f"{case[0]}-{case[1]}")
def decimation(request):
    """One `_decimate` call with both references, on the same cleaned input."""
    name, target_reduction = request.param
    vertices, faces = request.getfixturevalue(name)
    clean_vertices, clean_faces = remove_repeated_vertex_faces((vertices, faces))
    reference_points, reference_faces, collapses = simplify(
        clean_vertices,
        clean_faces,
        agg=7,
        target_reduction=target_reduction,
        return_collapses=True,
    )
    replay = replay_simplification(clean_vertices, clean_faces, collapses)
    got = _decimate(vertices, faces, 7, target_reduction)
    return (
        got,
        (reference_points, reference_faces),
        replay,
        (clean_vertices, clean_faces),
    )


def test_decimation_returns_simplify_mesh(decimation):
    """`simplify` is the reference for the mesh: the same points and faces."""
    (points, faces, _), (reference_points, reference_faces), _, _ = decimation
    np.testing.assert_array_equal(points, reference_points.astype(points.dtype))
    np.testing.assert_array_equal(faces, reference_faces)


@pytest.mark.parametrize(
    "vertex_dtype, face_dtype",
    [(np.float32, np.uint32), (np.float32, np.int32), (np.float64, np.int64)],
)
def test_decimation_keeps_the_input_dtypes(mesh, vertex_dtype, face_dtype):
    vertices, faces = mesh
    points, new_faces, _ = _decimate(
        vertices.astype(vertex_dtype), faces.astype(face_dtype), 7, 0.7
    )
    assert points.dtype == vertex_dtype
    assert new_faces.dtype == face_dtype


def test_decimation_mapping_matches_replay(decimation):
    """The replay is the reference for the mapping, up to its vertex numbering.

    Every vertex both map lands on the same output vertex, which a one-to-one
    translation between the two numberings shows, at the same position up to
    float32 rounding. Every vertex the replay drops is dropped here too.

    The converse does not hold. A piece that decimates to lines reaching no
    face is dropped here, but the replay sends its vertices to the point
    numbered just before it: `mapping[ip:] -= 1` shifts the dropped point
    `ip` itself onto `ip - 1`.
    """
    (points, _, mapping), _, (replay_points, _, replay_mapping), _ = decimation
    assert not np.any((replay_mapping < 0) & (mapping >= 0))

    both = (mapping >= 0) & (replay_mapping >= 0)
    pairs = np.unique(np.stack([mapping[both], replay_mapping[both]], axis=1), axis=0)
    assert len(np.unique(pairs[:, 0])) == len(pairs)
    assert len(np.unique(pairs[:, 1])) == len(pairs)

    gap = np.linalg.norm(
        points[mapping[both]] - replay_points[replay_mapping[both]], axis=1
    )
    assert gap.max() < 1.0


def test_decimation_drops_whole_components(decimation):
    """A dropped vertex takes its whole connected piece of the input with it."""
    (_, _, mapping), _, _, (vertices, faces) = decimation
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    n = len(vertices)
    adjacency = coo_array(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)
    )
    _, labels = connected_components(adjacency, directed=False)
    dropped = mapping < 0
    n_labels = labels.max() + 1
    dropped_per_label = np.bincount(labels, weights=dropped, minlength=n_labels)
    size_per_label = np.bincount(labels, minlength=n_labels)
    assert np.all((dropped_per_label == 0) | (dropped_per_label == size_per_label))


def test_decimation_drops_a_strip_that_collapses_to_a_line(mesh_with_collapsing_strip):
    vertices, faces = mesh_with_collapsing_strip
    _, _, mapping = _decimate(vertices, faces, 7, 0.7)
    assert np.all(mapping[-20:] == -1)
    assert np.all(mapping[:-20] >= 0)


def test_decimation_drops_an_unreferenced_vertex(mesh_with_unreferenced_vertex):
    vertices, faces = mesh_with_unreferenced_vertex
    _, _, mapping = _decimate(vertices, faces, 7, 0.7)
    assert mapping[unreferenced(vertices, faces)] == -1


def test_simplify_to_density_keeps_a_dropped_vertex_dropped(
    mesh_with_unreferenced_vertex, capsys
):
    """A -1 from one pass is not read as the last vertex of the next pass."""
    vertices, faces = mesh_with_unreferenced_vertex
    target = vertex_density((vertices, faces)) * 0.1
    new_vertices, _, mapping = simplify_to_density(
        (vertices, faces), target_density=target, tolerance=0.0, verbose=True
    )
    assert capsys.readouterr().out.count("iter") >= 2
    assert mapping[unreferenced(vertices, faces)] == -1
    assert mapping.max() < len(new_vertices)


# --- determinism (TASK-15) ------------------------------------------------

#: Run in a fresh interpreter, one mesh per invocation, printing a hash of the
#: simplified mesh and its mapping.
#:
#: A subprocess and not a loop in this process. The bug this guards against
#: lived in the *first* `fast_simplification.simplify` call of a process:
#: `fast-simplification` before 0.1.9 let its decimator read state left by a
#: previous call, so on a cold module state it read memory nothing had
#: written. Calls after the first were already stable, so an in-process loop
#: passed against the broken version and proved nothing.
_DETERMINISM_SCRIPT = """
import hashlib
import sys

import numpy as np
import pyvista as pv

from meshmash import simplify_mesh, simplify_to_density
from meshmash.utils import poly_to_mesh


def build(name):
    if name == "sphere":
        poly = pv.Sphere(radius=1000.0, theta_resolution=60, phi_resolution=60)
    else:
        poly = pv.ParametricTorus(ringradius=1000.0, crosssectionradius=350.0)
    vertices, faces = poly_to_mesh(poly.triangulate())
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces)


def digest(*arrays):
    running = hashlib.sha1()
    for array in arrays:
        running.update(np.ascontiguousarray(array).tobytes())
    return running.hexdigest()


mesh = build(sys.argv[1])
(vertices, faces), mapping = simplify_mesh(mesh, agg=7, target_reduction=0.7)
print("reduction", len(vertices), digest(vertices, faces, mapping))

vertices, faces, mapping = simplify_to_density(
    mesh, target_density=1e-5, simplify_agg=7
)
print("density", len(vertices), digest(vertices, faces, mapping))
"""


@pytest.mark.parametrize("mesh_name", ["sphere", "torus"])
def test_simplification_is_deterministic_from_a_cold_process(mesh_name):
    """Both simplifiers give the same bytes in a fresh interpreter, twice.

    `fast-simplification` below 0.1.9 fails this: three cold runs on the
    sample dendrite gave three different vertex arrays, and on the 2.3M-vertex
    neuron sample they gave three different vertex *counts*. The floor in
    pyproject.toml is what keeps this passing.
    """
    runs = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", _DETERMINISM_SCRIPT, mesh_name],
            capture_output=True,
            text=True,
            check=True,
        )
        runs.append(completed.stdout.strip())

    assert "reduction" in runs[0] and "density" in runs[0]
    assert runs[0] == runs[1], (
        f"two cold processes disagree for {mesh_name}:\n{runs[0]}\n{runs[1]}"
    )
