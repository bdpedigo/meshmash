import numpy as np
import pyvista as pv
from scipy.sparse import csr_array

from .types import Mesh


def mesh_to_poly(mesh: Mesh) -> pv.PolyData:
    if isinstance(mesh, pv.PolyData):
        return mesh
    elif isinstance(mesh, tuple):
        return pv.make_tri_mesh(*mesh)
    elif hasattr(mesh, "polydata"):
        return mesh.polydata
    elif hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        return pv.make_tri_mesh(mesh.vertices, mesh.faces)
    else:
        raise ValueError("Invalid mesh input.")


def mesh_to_adjacency(mesh: Mesh) -> csr_array:
    # TODO only use here because this is faster than numpy unique for unique extracting
    # edges, should be some other way to do this
    poly = mesh_to_poly(mesh)
    edge_data = poly.extract_all_edges(use_all_points=True, clear_data=True)
    lines = edge_data.lines
    edges = lines.reshape(-1, 3)[:, 1:]
    vertices = poly.points
    n_vertices = len(vertices)

    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)

    # was having some issue here with dijkstra, so casting to intc
    # REF: https://github.com/scipy/scipy/issues/20904
    # REF: related https://github.com/scipy/scipy/issues/20817
    adj = csr_array(
        (edge_lengths, (edges[:, 0].astype(np.intc), edges[:, 1].astype(np.intc))),
        shape=(n_vertices, n_vertices),
    )

    return adj


def poly_to_mesh(poly: pv.PolyData) -> Mesh:
    vertices = np.asarray(poly.points)
    faces = poly.faces.reshape(-1, 4)[:, 1:]
    return vertices, faces


def fix_mesh(mesh, **kwargs):
    from pymeshfix import MeshFix

    poly = mesh_to_poly(mesh)
    mesh_fix = MeshFix(poly)
    mesh_fix.repair(**kwargs)
    return poly_to_mesh(mesh_fix.mesh)
