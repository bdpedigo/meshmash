import numpy as np
from pymeshlab import Mesh as PyMesh
from pymeshlab import MeshSet

from .types import Mesh


def wrap_mesh(
    input_mesh: tuple,
    alpha: float = 250.0,
    offset: float = 5.0,
    alpha_fraction: float = None,
    offset_fraction: float = None,
) -> Mesh:
    """
    Parameters
    ----------
    alpha : float, optional
        Ball size in physical (mesh) units. If provided, overrides alpha_fraction.
    offset : float, optional
        Offset distance in physical (mesh) units. If provided, overrides offset_fraction.
    alpha_fraction : float
        Ball size as a fraction of the largest bounding-box diagonal (used when alpha is None).
    offset_fraction : float
        Offset distance as a fraction of the largest bounding-box diagonal (used when offset is None).
    """
    vertices = input_mesh[0]
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    largest_diagonal = np.linalg.norm(bbox_max - bbox_min)

    if alpha is not None:
        alpha_fraction = alpha / largest_diagonal
    if offset is not None:
        offset_fraction = offset / largest_diagonal

    ms = MeshSet()
    mesh = PyMesh(*input_mesh)
    ms.add_mesh(mesh)

    # TODO update to use the more modern PyMeshLab parameters
    _ = ms.generate_alpha_wrap(
        alpha_fraction=alpha_fraction, offset_fraction=offset_fraction
    )

    back = ms.current_mesh()

    vertices = back.vertex_matrix().astype(input_mesh[0].dtype)
    faces = back.face_matrix().astype(input_mesh[1].dtype)

    return (vertices, faces)
