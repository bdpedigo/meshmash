# Some contents of this file adapted from:
# https://github.com/RobinMagnet/pyFM/blob/master/pyFM/mesh/laplacian.py
# (MIT License)
import numpy as np
import scipy.sparse as sparse

from .types import Mesh, interpret_mesh


def area_matrix(mesh: Mesh) -> sparse.dia_matrix:
    """
    Compute the diagonal matrix of lumped vertex area for mesh laplacian.
    Entry i on the diagonal is the area of vertex i, approximated as one third
    of adjacent triangles

    Parameters
    -----------------------------
    vertices   :
        (n,3) array of vertices coordinates
    faces      :
        (m,3) array of vertex indices defining faces
    Returns
    -----------------------------
    M :
        (n,n) sparse diagonal matrix of vertex areas in dia format
    """
    vertices, faces = interpret_mesh(mesh)
    N = vertices.shape[0]

    # Compute face area
    v1 = vertices[faces[:, 0]]  # (m,3)
    v2 = vertices[faces[:, 1]]  # (m,3)
    v3 = vertices[faces[:, 2]]  # (m,3)
    faces_areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)  # (m,)

    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    J = np.zeros_like(I)
    V = np.concatenate([faces_areas, faces_areas, faces_areas]) / 3

    # Get array of vertex areas
    vertex_areas = np.array(
        sparse.coo_matrix((V, (I, J)), shape=(N, 1)).todense()
    ).flatten()

    A = sparse.dia_matrix((vertex_areas, 0), shape=(N, N))
    return A


def _cotangent_laplacian(mesh: Mesh) -> sparse.csc_matrix:
    """
    Compute the cotangent weights matrix for mesh laplacian.

    Parameters
    -----------------------------
    vertices   :
        (n,3) array of vertices coordinates
    faces      :
        (m,3) array of vertex indices defining faces
    faces_area :
        (m,) - Optional, array of per-face area

    Returns
    -----------------------------
    A : scipy.sparse.csc_matrix
        (n,n) sparse area matrix in csc format
    """
    vertices, faces = interpret_mesh(mesh)
    N = vertices.shape[0]

    v1 = vertices[faces[:, 0]]  # (m,3)
    v2 = vertices[faces[:, 1]]  # (m,3)
    v3 = vertices[faces[:, 2]]  # (m,3)

    # Edge lengths indexed by opposite vertex
    u1 = v3 - v2
    u2 = v1 - v3
    u3 = v2 - v1

    L1 = np.linalg.norm(u1, axis=1)  # (m,)
    L2 = np.linalg.norm(u2, axis=1)  # (m,)
    L3 = np.linalg.norm(u3, axis=1)  # (m,)

    # Compute cosine of angles
    A1 = np.einsum("ij,ij->i", -u2, u3) / (L2 * L3)  # (m,)
    A2 = np.einsum("ij,ij->i", u1, -u3) / (L1 * L3)  # (m,)
    A3 = np.einsum("ij,ij->i", -u1, u2) / (L1 * L2)  # (m,)

    # Use cot(arccos(x)) = x/sqrt(1-x^2)
    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    S = np.concatenate([A3, A1, A2])
    S = 0.5 * S / np.sqrt(1 - S**2)

    In = np.concatenate([I, J, I, J])
    Jn = np.concatenate([J, I, I, J])
    Sn = np.concatenate([-S, -S, S, S])

    L = sparse.csc_matrix((Sn, (In, Jn)), shape=(N, N))
    return L


def cotangent_laplacian(
    mesh: Mesh, robust=False, mollify_factor=1e-5
) -> tuple[sparse.csc_array, sparse.dia_array]:
    if robust:
        try:
            from robust_laplacian import mesh_laplacian
        except (ImportError, ModuleNotFoundError) as e:
            msg = (
                "To use the robust laplacian features, please install the "
                "'robust-laplacian' package."
            )
            raise e.__class__(msg) from e

        vertices = mesh[0]
        faces = mesh[1]
        vertices = np.asarray(vertices)
        faces = np.asarray(faces)
        L, M = mesh_laplacian(vertices, faces, mollify_factor=mollify_factor)
        L = sparse.csc_array(L)
        M = sparse.dia_array(M)
        return L, M
    else:
        L = _cotangent_laplacian(mesh)
        M = area_matrix(mesh)
        return L, M


def compute_vertex_areas(mesh, robust=False, mollify_factor=1e-5):
    if not robust:
        return area_matrix(mesh).diagonal()
    else:
        _, D = cotangent_laplacian(mesh, robust=robust, mollify_factor=mollify_factor)
        return D.diagonal()
