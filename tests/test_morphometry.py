import numpy as np
import pyvista as pv
from gpytoolbox import fast_winding_number
from point_cloud_utils import closest_points_on_mesh

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


def test_rep_point_is_inside_and_deepest():
    mesh = _two_spheres()
    labels = np.zeros(len(mesh[0]), dtype=np.int32)

    results, _ = component_morphometry_pipeline(mesh, labels, select_label=0, seed=0)

    vertices, faces = mesh
    # point_cloud_utils wants the points and vertices in one dtype and memory order.
    rep = np.ascontiguousarray(results[["rep_x", "rep_y", "rep_z"]], dtype=vertices.dtype)
    medoid = np.ascontiguousarray(results[["x", "y", "z"]], dtype=vertices.dtype)
    vertices = np.ascontiguousarray(vertices)
    assert results["rep_x"].dtype == np.float32
    assert (fast_winding_number(rep, vertices, faces) > 0.5).all()
    rep_depth, _, _ = closest_points_on_mesh(rep, vertices, faces)
    medoid_depth, _, _ = closest_points_on_mesh(medoid, vertices, faces)
    # The rep point is where max_dt_nm is measured, so no sample is deeper.
    np.testing.assert_allclose(rep_depth, results["max_dt_nm"], rtol=1e-4)
    assert (rep_depth >= medoid_depth - 1e-3).all()
