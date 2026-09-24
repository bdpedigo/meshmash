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
