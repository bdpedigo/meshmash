import numpy as np
from scipy.sparse import csr_array, diags_array, eye_array
from scipy.sparse.linalg import spsolve


def to_laplacian(adj):
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


def to_adjacency(vertices, edges):
    """
    Convert edges and vertices to a sparse adjacency matrix.
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


def label_propagation(adjacency, labels, alpha=0.995):
    # run label propagation
    laplacian = to_laplacian(adjacency)
    identity = eye_array(laplacian.shape[0])
    invertee = identity - alpha * laplacian
    F = spsolve(invertee, labels)
    out = np.squeeze((1 - alpha) * F)

    return out

    # # look at the ratio for excitatory
    # ratio = out[:, 0] / (out.sum(axis=1))
