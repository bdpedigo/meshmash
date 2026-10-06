import numpy as np
import pyvista as pv

from meshmash.pipelines.morphometry import component_morphometry_pipeline
from meshmash.utils import poly_to_mesh


def test_n_post_synapses_is_int32():
    # uint16 would wrap silently past 65,535 synapses on one structure.
    poly = pv.Sphere(radius=1000.0, theta_resolution=24, phi_resolution=24)
    vertices, faces = poly_to_mesh(poly.triangulate())
    mesh = (np.asarray(vertices), np.asarray(faces))
    labels = np.zeros(len(vertices), dtype=np.int32)

    results, _ = component_morphometry_pipeline(
        mesh, labels, select_label=0, post_synapse_mappings=np.arange(5)
    )

    assert results["n_post_synapses"].dtype == np.int32
    assert results["n_post_synapses"].sum() == 5


def _two_spheres():
    left = pv.Sphere(radius=1000.0, theta_resolution=24, phi_resolution=24)
    right = left.translate((5000.0, 0.0, 0.0), inplace=False)
    vertices, faces = poly_to_mesh((left + right).triangulate())
    return np.asarray(vertices), np.asarray(faces)


def test_seed_makes_the_estimates_reproducible():
    mesh = _two_spheres()
    labels = np.zeros(len(mesh[0]), dtype=np.int32)

    runs = [
        component_morphometry_pipeline(mesh, labels, select_label=0, seed=3)[0]
        for _ in range(5)
    ]

    columns = [c for c in runs[0].columns if c != "time"]
    assert len(runs[0]) == 2
    assert all(runs[0][columns].equals(run[columns]) for run in runs[1:])


def test_synapses_are_counted_per_component():
    mesh = _two_spheres()
    labels = np.zeros(len(mesh[0]), dtype=np.int32)
    n_left = len(mesh[0]) // 2

    results, components = component_morphometry_pipeline(
        mesh,
        labels,
        select_label=0,
        post_synapse_mappings=np.array([0, 1, 2, n_left]),
        seed=0,
    )

    assert results.loc[components[0], "n_post_synapses"] == 3
    assert results.loc[components[n_left], "n_post_synapses"] == 1


def test_no_measured_component_marks_every_vertex_unmeasured():
    mesh = _two_spheres()
    labels = np.zeros(len(mesh[0]), dtype=np.int32)

    results, components = component_morphometry_pipeline(mesh, labels, select_label=1)

    assert results.empty
    assert components.dtype == np.int32
    assert (components == -1).all()
