import subprocess
import sys

import numpy as np
import pytest

from meshmash import simplify_mesh


@pytest.fixture(scope="module")
def simplified(mesh):
    return simplify_mesh(mesh, target_reduction=0.7)


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
    """The reason both come from replay: the other ordering does not agree.

    Remapping the input's own faces through the mapping has to land on real
    vertices of the mesh that came back, which is exactly what fails if the
    mesh is taken from ``simplify`` and the mapping from
    ``replay_simplification``.
    """
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
