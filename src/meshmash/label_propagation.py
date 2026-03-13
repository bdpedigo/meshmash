from typing import Union

import numpy as np
from scipy.sparse import csr_array, diags_array, eye_array
from scipy.sparse.linalg import spsolve


def to_laplacian(adj: Union[csr_array, np.ndarray]) -> csr_array:
    adj = csr_array(adj)
    # adj[adj > 0] = 1
    # adj[np.arange(adj.shape[0]), np.arange(adj.shape[0])] = 0
    # adj.eliminate_zeros()
    adj = adj + adj.T
    degrees = adj.sum(axis=1)
    degree_norm = 1 / np.sqrt(degrees)
    D = diags_array(degree_norm)
    lap = D @ adj @ D

    return lap


def to_adjacency(vertices: np.ndarray, edges: np.ndarray) -> csr_array:
    """Convert an edge list to a sparse CSR adjacency matrix.

    Parameters
    ----------
    vertices :
        Array of vertex positions used only to determine the matrix size,
        shape ``(V, d)``.
    edges :
        Array of directed edge pairs, shape ``(E, 2)``.

    Returns
    -------
    :
        Unweighted sparse adjacency matrix of shape ``(V, V)``.
    """
    # edges = edges[edges[:, 0] != edges[:, 1]]  # remove self-loops
    row_ind = edges[:, 0].astype(np.intc)
    col_ind = edges[:, 1].astype(np.intc)
    adjacency = csr_array(
        (
            np.ones(len(row_ind)),
            (row_ind, col_ind),
        ),  # data, row indices, column indices),
        shape=(len(vertices), len(vertices)),
    )
    return adjacency


def label_propagation(
    adjacency: csr_array, labels: np.ndarray, alpha: float = 0.995
) -> np.ndarray:
    """Propagate labels across a graph using a closed-form diffusion solution.

    Solves the linear system
    :math:`F = (1 - \\alpha)(I - \\alpha \\hat{L})^{-1} Y`
    where :math:`\\hat{L}` is the symmetric normalised graph Laplacian and
    :math:`Y` contains the initial label scores.

    Parameters
    ----------
    adjacency :
        Sparse adjacency matrix of the graph, shape ``(V, V)``.
    labels :
        Initial label score array of shape ``(V,)`` or ``(V, K)`` for
        ``K`` label classes.  Use 0 for unlabelled vertices.
    alpha :
        Damping factor in ``[0, 1)``.  Values close to 1 allow labels to
        diffuse far from their source; values close to 0 keep label scores
        near the initialisation.

    Returns
    -------
    :
        Smoothed label score array with the same shape as ``labels``.
    """
    # run label propagation
    laplacian = to_laplacian(adjacency)
    identity = eye_array(laplacian.shape[0])
    invertee = identity - alpha * laplacian
    F = spsolve(invertee, labels)
    out = np.squeeze((1 - alpha) * F)

    return out

    # # look at the ratio for excitatory
    # ratio = out[:, 0] / (out.sum(axis=1))
