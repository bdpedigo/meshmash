import logging
import time
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from joblib import Parallel, delayed
from scipy.sparse import csr_array, diags_array
from scipy.sparse.csgraph import connected_components, dijkstra, laplacian
from scipy.sparse.linalg import eigsh
from scipy.stats import rankdata
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

from .decompose import decompose_laplacian
from .laplacian import cotangent_laplacian
from .types import Mesh, interpret_mesh
from .utils import (
    mesh_to_adjacency,
    mesh_to_poly,
    rough_subset_mesh_by_indices,
    subset_mesh_by_indices,
)


def graph_laplacian_split(
    adj: csr_array, dtype: type = np.float32
) -> tuple[np.ndarray, np.ndarray]:
    """Bisect a mesh using the Fiedler vector of the graph Laplacian.

    Computes the second smallest eigenvector (Fiedler vector) of the
    unnormalised graph Laplacian and partitions vertices by the sign of
    their Fiedler coefficient.

    Parameters
    ----------
    adj :
        Sparse adjacency matrix of the mesh graph, shape ``(V, V)``.
    dtype :
        Floating-point dtype used for the eigensolver.

    Returns
    -------
    indices1 :
        Indices of vertices in the first partition
        (Fiedler coefficient >= 0).
    indices2 :
        Indices of vertices in the second partition
        (Fiedler coefficient < 0).
    """
    # probably some issue with tolerance/sigma?
    # TODO normed didn't seem to make much of a difference here; perhaps just because
    # degrees are fairly homogeneous?
    lap = laplacian(adj, normed=False, symmetrized=True, return_diag=False, dtype=dtype)
    if dtype == np.float32 or dtype == "float32":
        eigen_tol = 1e-7
    elif dtype == np.float64 or dtype == "float64":
        eigen_tol = 1e-10

    # TODO cannot figure out why this isn't deterministic
    # or if the random errors are from some other part of the pipeline
    # v0 = np.full(n, 1 / np.sqrt(n), dtype=dtype)
    eigenvalues, eigenvectors = eigsh(
        lap,
        k=2,
        sigma=-1e-10,
        # v0=v0, # dropped this to see if it fixes the issue with failing splits
        tol=eigen_tol,
        maxiter=20,
        ncv=20,  # TODO revisit this sensitivity to NCV for speed
    )

    index = np.argmax(eigenvalues)
    eigenvector = eigenvectors[:, index]
    eigenvector *= np.sign(eigenvector[0]) * 1
    indices1 = np.nonzero(eigenvector >= 0)[0]
    indices2 = np.nonzero(eigenvector < 0)[0]

    return indices1, indices2


def bisect_adjacency(
    adj: csr_array, n_retries: int = 7, check: bool = True
) -> tuple[tuple[csr_array, csr_array], tuple[np.ndarray, np.ndarray]]:
    """Bisect a mesh graph into two parts using the graph-Laplacian Fiedler vector.

    Calls [graph_laplacian_split][meshmash.split.graph_laplacian_split] and retries if the result is
    degenerate (one empty partition or disconnected nodes).  Uses
    recursion up to ``n_retries`` times.

    Parameters
    ----------
    adj :
        Sparse adjacency matrix of shape ``(V, V)``.
    n_retries :
        Maximum number of retry attempts when the split fails to produce
        two non-empty, connected partitions.
    check :
        If ``True``, verify that no vertex becomes isolated (zero-degree)
        after splitting and retry if so.

    Returns
    -------
    sub_adjs :
        Pair of sub-adjacency matrices ``(adj1, adj2)`` for each partition.
    submesh_indices :
        Pair of index arrays ``(indices1, indices2)`` mapping each
        partition's rows back to the original ``adj``.
    """
    if n_retries == 0:
        logging.info("Adjacency shape: %s", adj.shape)
        logging.info("Adjacency nnz: %s", adj.nnz)
        raise RuntimeError("Split failed to divide mesh.")

    # get the split indices
    indices1, indices2 = graph_laplacian_split(adj)

    if len(indices1) == 0 or len(indices2) == 0:
        # print(adj.shape)
        logging.info("Split failed to divide mesh, retrying.")
        return bisect_adjacency(adj, n_retries=n_retries - 1)

    # get the sub-adjacencies
    sub_adj1 = adj[indices1][:, indices1]
    sub_adj2 = adj[indices2][:, indices2]

    if check:
        # make sure we didn't disconnect any nodes
        degrees1 = np.sum(sub_adj1, axis=1) + np.sum(sub_adj1, axis=0)
        degrees2 = np.sum(sub_adj2, axis=1) + np.sum(sub_adj2, axis=0)
        if np.any(degrees1 == 0) or np.any(degrees2 == 0):
            # TODO no idea why retrying here helps almost always after one go...
            # did not think randomness should have that much of an effect?
            logging.info("Some nodes were disconnected in the split, retrying.")
            return bisect_adjacency(adj, n_retries=n_retries - 1)

    sub_adjs = (sub_adj1, sub_adj2)
    submesh_indices = (indices1, indices2)

    return sub_adjs, submesh_indices


def _interpret_adjacency(mesh: Union[Mesh, np.ndarray, csr_array]) -> csr_array:
    """Return a mesh graph's adjacency, accepting either a mesh or an adjacency.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh],
        or an adjacency matrix that is already one of those.

    Returns
    -------
    :
        Sparse adjacency matrix of the mesh graph, shape ``(V, V)``.
    """
    if isinstance(mesh, (csr_array, np.ndarray)):
        return mesh
    return mesh_to_adjacency(interpret_mesh(mesh))


def _order_split_by_size(submesh_mapping: np.ndarray) -> np.ndarray:
    """Relabel a split so that label ``0`` is the largest chunk.

    Parameters
    ----------
    submesh_mapping :
        Per-vertex integer label array, ``-1`` for a vertex in no chunk.

    Returns
    -------
    :
        The same partition with labels renumbered ``0, 1, …, K-1`` from
        largest chunk to smallest.  ``-1`` stays ``-1``.
    """
    valid_submesh_mapping = submesh_mapping[submesh_mapping != -1]
    if len(valid_submesh_mapping) == 0:
        # Every connected component fell under min_vertex_threshold, so
        # there is nothing to reorder and no label to be the largest.
        # Returning the all -1 mapping leaves that for the caller to
        # notice; the reordering below raises on the empty maximum.
        return submesh_mapping
    labels, counts = np.unique(valid_submesh_mapping, return_counts=True)
    reorder = np.argsort(-counts)
    new_labels = np.arange(labels.max() + 1)
    old_to_new = dict(zip(labels[reorder], new_labels))
    old_to_new[-1] = -1
    return np.vectorize(old_to_new.get)(submesh_mapping)


def _fit_split_by_queue(
    whole_adj: csr_array,
    make_piece: Callable[[np.ndarray], Any],
    size_of: Callable[[Any], int],
    cut: Callable[[Any], tuple[Sequence[Any], Sequence[np.ndarray]]],
    max_vertex_threshold: int = 20_000,
    min_vertex_threshold: int = 100,
    max_rounds: int = 100_000,
    verbose: Union[bool, int] = False,
) -> np.ndarray:
    """Run a cut over a work queue until every piece is small enough.

    Every way of splitting a mesh in this module is this one loop with a
    different cut in the middle: seed the queue with the connected components
    worth keeping, pull a piece off, cut it if it is too big, and put the
    sub-pieces back.  The loop lives here so that a new cut method is a cut and
    nothing else.  The component pre-pass, the index bookkeeping, the runaway
    guard, the ``-1`` convention and the largest-first ordering are shared, so
    a method cannot get one of them subtly wrong on its own.

    A *piece* is whatever state the cut works on.  It is opaque here: the graph
    Laplacian cut carries a sub-adjacency, the cotangent cut carries an
    ``(L, M)`` pair.  Each cut slices its own piece out of its parent rather
    than out of the whole mesh, which is why the piece travels through the
    queue at all.

    Parameters
    ----------
    whole_adj :
        Sparse adjacency matrix of the whole mesh graph, shape ``(V, V)``.
        Used for the connected-component pre-pass and for the vertex count.
    make_piece :
        Builds the piece for one connected component, given that component's
        vertex indices into the whole mesh.
    size_of :
        Returns the number of vertices a piece holds.
    cut :
        Splits one piece into two or more.  Returns the sub-pieces, and for
        each sub-piece the indices it occupies *within its parent piece*.
        Raise from inside ``cut`` if a piece cannot be divided.
    max_vertex_threshold :
        Stop cutting a piece once it contains at most this many vertices.
    min_vertex_threshold :
        Discard connected components with fewer than this many vertices;
        their vertices receive label ``-1``.
    max_rounds :
        Maximum number of cuts before the loop stops regardless of the
        remaining piece sizes.
    verbose :
        If truthy, print the queue size every 50 rounds.

    Returns
    -------
    :
        Per-vertex integer label array of shape ``(V,)``.  Labels run
        ``0, 1, …, K-1`` ordered from largest to smallest chunk; vertices
        not assigned to any chunk have label ``-1``.
    """
    n_vertices = whole_adj.shape[0]
    mesh_indices = np.arange(n_vertices)

    # first, append all the connected components that are large enough to the queue
    n_components, component_labels = connected_components(whole_adj)

    queue = []
    for component_id in range(n_components):
        component_mask = component_labels == component_id
        count = component_mask.sum()
        if count >= min_vertex_threshold:
            component_indices = mesh_indices[component_mask]
            queue.append((make_piece(component_indices), component_indices))

    submesh_mapping = np.full(n_vertices, -1, dtype=int)

    n_finished = 0
    rounds = 0

    while len(queue) > 0 and rounds < max_rounds:
        if verbose and rounds % 50 == 0:
            print("Meshes in queue:", len(queue))
        current_piece, current_indices = queue.pop(0)

        # if this piece is small enough, it is finished as it stands; this
        # happens when a connected component is already small
        if size_of(current_piece) <= max_vertex_threshold:
            pieces, indices_to_main = [current_piece], [current_indices]
        else:  # otherwise, cut it
            pieces, local_indices = cut(current_piece)

            # adjust indices to be in terms of the main mesh
            indices_to_main = [current_indices[indices] for indices in local_indices]

        for piece, indices in zip(pieces, indices_to_main):
            if size_of(piece) > max_vertex_threshold:
                queue.append((piece, indices))
            else:
                submesh_mapping[indices] = n_finished
                n_finished += 1
        rounds += 1

    # remap so the first submesh is largest
    return _order_split_by_size(submesh_mapping)


def fit_mesh_split(
    mesh: Union[Mesh, np.ndarray, csr_array],
    max_vertex_threshold: int = 20_000,
    min_vertex_threshold: int = 100,
    max_rounds: int = 100_000,
    verbose: Union[bool, int] = False,
) -> np.ndarray:
    """Partition a mesh into non-overlapping chunks using recursive bisection.

    Repeatedly bisects each chunk using the Fiedler vector of the graph
    Laplacian until every chunk has at most ``max_vertex_threshold`` vertices.
    Chunks belonging to connected components with fewer than
    ``min_vertex_threshold`` vertices are dropped (their vertices receive
    label ``-1``).  The returned labels are ordered so that the largest
    chunk has label ``0``.

    Parameters
    ----------
    mesh :
        Input mesh, adjacency matrix, or vertex array accepted by
        [interpret_mesh][meshmash.types.interpret_mesh] /
        [mesh_to_adjacency][meshmash.utils.mesh_to_adjacency].
    max_vertex_threshold :
        Stop bisecting a chunk once it contains at most this many vertices.
    min_vertex_threshold :
        Discard connected components with fewer than this many vertices;
        their vertices receive label ``-1``.
    max_rounds :
        Maximum number of bisection steps before the algorithm terminates
        regardless of remaining chunk sizes.
    verbose :
        If truthy, print queue size every 50 rounds.

    Returns
    -------
    :
        Per-vertex integer label array of shape ``(V,)``.  Labels run
        ``0, 1, …, K-1`` ordered from largest to smallest chunk; vertices
        not assigned to any chunk have label ``-1``.
    """
    whole_adj = _interpret_adjacency(mesh)

    return _fit_split_by_queue(
        whole_adj,
        lambda indices: whole_adj[indices][:, indices],
        lambda adj: adj.shape[0],
        bisect_adjacency,
        max_vertex_threshold=max_vertex_threshold,
        min_vertex_threshold=min_vertex_threshold,
        max_rounds=max_rounds,
        verbose=verbose,
    )


def subset_diags(matrix: sparse.sparray, indices: np.ndarray) -> diags_array:
    return diags_array(matrix.diagonal()[indices], shape=(len(indices), len(indices)))


def bisect_laplacian(
    L: sparse.sparray, M: sparse.sparray
) -> tuple[
    tuple[tuple[sparse.sparray, diags_array], tuple[sparse.sparray, diags_array]],  # noqa: E501
    tuple[np.ndarray, np.ndarray],
]:
    """Bisect a mesh into two parts using the cotangent-Laplacian Fiedler vector.

    Computes the second eigenvector of the generalised eigenproblem
    ``L v = λ M v`` via [decompose_laplacian][meshmash.decompose.decompose_laplacian]
    and partitions vertices by its sign.

    Parameters
    ----------
    L :
        Cotangent Laplacian matrix of shape ``(V, V)`` as returned by
        [cotangent_laplacian][meshmash.laplacian.cotangent_laplacian].
    M :
        Diagonal mass matrix of shape ``(V, V)`` as returned by
        [cotangent_laplacian][meshmash.laplacian.cotangent_laplacian].

    Returns
    -------
    sub_laps :
        Pair of ``(L_sub, M_sub)`` tuples for each partition, where
        ``M_sub`` is a [diags_array][scipy.sparse.diags_array] of the diagonal
        mass entries for that partition.
    submesh_indices :
        Pair of index arrays ``(indices1, indices2)`` mapping each
        partition's rows back to the original ``L``.
    """
    # get the split indices
    # indices1, indices2 = graph_laplacian_split(adj)

    _, eigenvectors = decompose_laplacian(L, M, n_components=2)
    indices1 = np.nonzero(eigenvectors[:, 1] >= 0)[0]
    indices2 = np.nonzero(eigenvectors[:, 1] < 0)[0]

    # get the sub-adjacencies
    sub_adj1 = L[indices1][:, indices1]
    sub_adj2 = L[indices2][:, indices2]

    sub_laps = (
        (sub_adj1, subset_diags(M, indices1)),
        (sub_adj2, subset_diags(M, indices2)),
    )

    submesh_indices = (indices1, indices2)

    return sub_laps, submesh_indices


def fit_mesh_split_lap(
    mesh: Mesh,
    max_vertex_threshold: int = 20_000,
    min_vertex_threshold: int = 100,
    max_rounds: int = 100_000,
    robust: bool = True,
    mollify_factor: float = 1e-5,
    verbose: Union[bool, int] = False,
) -> np.ndarray:
    """Partition a mesh into chunks using recursive cotangent-Laplacian bisection.

    Like [fit_mesh_split][meshmash.split.fit_mesh_split] but uses the Fiedler vector of the
    cotangent Laplacian (instead of the graph Laplacian) for each
    bisection step, which can produce more geometrically uniform
    partitions on irregular meshes.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
        An adjacency matrix is not enough here, unlike in
        [fit_mesh_split][meshmash.split.fit_mesh_split]: the cotangent
        Laplacian is built from the vertex positions.
    max_vertex_threshold :
        Stop bisecting a chunk once it contains at most this many vertices.
    min_vertex_threshold :
        Discard connected components with fewer than this many vertices;
        their vertices receive label ``-1``.
    max_rounds :
        Maximum number of bisection steps before the algorithm terminates.
    robust :
        If ``True``, use the robust cotangent Laplacian variant; passed
        to [cotangent_laplacian][meshmash.laplacian.cotangent_laplacian].
    mollify_factor :
        Mollification factor passed to
        [cotangent_laplacian][meshmash.laplacian.cotangent_laplacian].
    verbose :
        If truthy, print queue information each round.

    Returns
    -------
    :
        Per-vertex integer label array of shape ``(V,)``.  Labels run
        ``0, 1, …, K-1`` ordered from largest to smallest chunk; vertices
        not assigned to any chunk have label ``-1``.
    """
    if isinstance(mesh, (csr_array, np.ndarray)):
        raise TypeError(
            "fit_mesh_split_lap needs the mesh itself, not an adjacency "
            "matrix: the cotangent Laplacian is built from vertex positions"
        )
    mesh = interpret_mesh(mesh)
    whole_adj = mesh_to_adjacency(mesh)

    def make_piece(indices: np.ndarray):
        submesh = subset_mesh_by_indices(mesh, indices)
        return cotangent_laplacian(
            submesh, robust=robust, mollify_factor=mollify_factor
        )

    return _fit_split_by_queue(
        whole_adj,
        make_piece,
        lambda piece: piece[0].shape[0],
        lambda piece: bisect_laplacian(*piece),
        max_vertex_threshold=max_vertex_threshold,
        min_vertex_threshold=min_vertex_threshold,
        max_rounds=max_rounds,
        verbose=verbose,
    )


def geodesic_voronoi_split(adj: csr_array, n_cells: int) -> np.ndarray:
    """Cut a mesh graph into ``n_cells`` geodesic Voronoi cells.

    Picks ``n_cells`` seed vertices by farthest-point sampling over
    edge-length distances, then gives every vertex to the seed that reaches
    it first along the surface.

    Two properties follow from the construction, and they are the reason to
    cut a mesh this way.  A cell of a connected graph is connected, because
    any vertex on a shortest path to the owning seed is itself owned by that
    seed.  And every vertex lies within the sampling radius of its own seed,
    so the cells are compact.  Compactness is what a distance-based collar such as
    [MeshStitcher.expand_split][meshmash.split.MeshStitcher.expand_split]
    prices.

    Each farthest-point step is a single-source
    [dijkstra][scipy.sparse.csgraph.dijkstra] truncated at the current
    sampling radius, since a new seed can only improve vertices nearer to it
    than that.  Seeding therefore costs far less than one full traversal per
    seed.

    The cut is deterministic and takes no random seed: the first seed is
    vertex ``0`` and ties break by index.

    Parameters
    ----------
    adj :
        Sparse adjacency matrix of the mesh graph, shape ``(V, V)``, weighted
        by edge length as [mesh_to_adjacency][meshmash.utils.mesh_to_adjacency]
        returns it.
    n_cells :
        Number of seeds, and so the largest number of cells.

    Returns
    -------
    :
        Per-vertex cell label of shape ``(V,)``, running ``0, 1, …``.  A cell
        can come back empty if two seeds land on the same vertex.
    """
    nearest = dijkstra(adj, directed=False, indices=[0], min_only=True)
    seeds = [0]
    for _ in range(1, n_cells):
        farthest = int(np.argmax(nearest))
        update = dijkstra(
            adj,
            directed=False,
            indices=[farthest],
            min_only=True,
            limit=float(nearest[farthest]),
        )
        nearest = np.minimum(nearest, update)
        seeds.append(farthest)

    _, _, sources = dijkstra(
        adj,
        directed=False,
        indices=seeds,
        min_only=True,
        return_predecessors=True,
    )
    lookup = np.full(adj.shape[0], -1, dtype=np.int64)
    lookup[np.asarray(seeds)] = np.arange(len(seeds))
    return lookup[sources]


def fit_mesh_split_geodesic(
    mesh: Union[Mesh, np.ndarray, csr_array],
    max_vertex_threshold: int = 20_000,
    min_vertex_threshold: int = 100,
    target_vertices: int = 10_000,
    max_rounds: int = 100_000,
    verbose: Union[bool, int] = False,
) -> np.ndarray:
    """Partition a mesh into non-overlapping chunks by geodesic Voronoi cells.

    An alternative to
    [fit_mesh_split][meshmash.split.fit_mesh_split] that never solves an
    eigenproblem.  A piece over ``max_vertex_threshold`` vertices is cut into
    ``ceil(n / target_vertices)`` cells at once by
    [geodesic_voronoi_split][meshmash.split.geodesic_voronoi_split], and a
    cell still over the threshold is cut again.  The recursion is what makes
    the chunk sizes ragged: a dense patch such as a soma packs more vertices
    into a sampling radius than a thin process does, so it gets re-cut.

    Where the two differ, beyond the cost of the cut itself.  Spectral
    bisection halves a piece by a Fiedler sign, which balances the two sides
    but says nothing about how far a chunk reaches across the surface.  This
    method bounds the reach instead and lets the sizes vary.  Chunks are
    connected by construction here, which the bisection does not promise, and
    the result is reproducible run to run, which the bisection is not.

    Parameters
    ----------
    mesh :
        Input mesh, adjacency matrix, or vertex array accepted by
        [interpret_mesh][meshmash.types.interpret_mesh] /
        [mesh_to_adjacency][meshmash.utils.mesh_to_adjacency].
    max_vertex_threshold :
        Stop cutting a chunk once it contains at most this many vertices.
    min_vertex_threshold :
        Discard connected components with fewer than this many vertices;
        their vertices receive label ``-1``.
    target_vertices :
        Wanted vertices per cell, which sets how many seeds a piece gets.
        Cells come out near this size, and always under
        ``max_vertex_threshold``.  Must be a positive integer.
    max_rounds :
        Maximum number of cuts before the algorithm terminates regardless of
        remaining chunk sizes.
    verbose :
        If truthy, print queue size every 50 rounds.

    Returns
    -------
    :
        Per-vertex integer label array of shape ``(V,)``.  Labels run
        ``0, 1, …, K-1`` ordered from largest to smallest chunk; vertices
        not assigned to any chunk have label ``-1``.

    Raises
    ------
    ValueError
        If ``target_vertices`` is not a positive integer.
    """
    if target_vertices < 1:
        raise ValueError(
            f"target_vertices must be a positive integer, got {target_vertices}"
        )

    whole_adj = _interpret_adjacency(mesh)

    def cut(adj: csr_array) -> tuple[list[csr_array], list[np.ndarray]]:
        n_cells = max(2, int(np.ceil(adj.shape[0] / target_vertices)))
        cell_of_vertex = geodesic_voronoi_split(adj, n_cells)
        pieces = []
        local_indices = []
        for cell in range(n_cells):
            members = np.nonzero(cell_of_vertex == cell)[0]
            if len(members) > 0:
                pieces.append(adj[members][:, members])
                local_indices.append(members)
        if len(pieces) < 2:
            raise RuntimeError(
                f"the geodesic cut left a {adj.shape[0]}-vertex piece whole; "
                "every seed landed on the same vertex"
            )
        return pieces, local_indices

    return _fit_split_by_queue(
        whole_adj,
        lambda indices: whole_adj[indices][:, indices],
        lambda adj: adj.shape[0],
        cut,
        max_vertex_threshold=max_vertex_threshold,
        min_vertex_threshold=min_vertex_threshold,
        max_rounds=max_rounds,
        verbose=verbose,
    )


def apply_mesh_split(mesh: Mesh, split_mapping: np.ndarray) -> list[Mesh]:
    """Separate a mesh into submeshes according to a per-vertex split mapping.

    Only faces whose *all three* vertices share the same label are retained
    in that submesh.  Faces that span label boundaries are dropped.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    split_mapping :
        Per-vertex integer label array of length ``V`` as produced by
        [fit_mesh_split][meshmash.split.fit_mesh_split].  Vertices with label ``-1`` are excluded.

    Returns
    -------
    :
        List of ``(vertices, faces)`` tuples, one per unique non-negative
        label, with remapped face indices.
    """
    vertices, faces = interpret_mesh(mesh)
    faces = pd.DataFrame(faces)
    faces["label0"] = split_mapping[faces[0]]
    faces["label1"] = split_mapping[faces[1]]
    faces["label2"] = split_mapping[faces[2]]
    faces = faces.query("label0 == label1 and label1 == label2")
    new_meshes = []
    for label, sub_faces in faces.groupby("label0"):
        if label == -1:
            continue
        sub_faces = sub_faces[[0, 1, 2]].values
        select_indices = np.unique(sub_faces)  # these should be ordered by original
        index_remapping = dict(zip(select_indices, np.arange(len(select_indices))))
        sub_faces = np.vectorize(index_remapping.get)(sub_faces)
        sub_points = vertices[select_indices]
        new_meshes.append((sub_points, sub_faces))
    return new_meshes


def get_submesh_borders(submesh: Mesh) -> np.ndarray:
    """Return the vertex indices that lie on boundary edges of a submesh.

    Uses :mod:`pyvista` boundary-edge extraction to find edges shared by
    fewer than two faces.

    Parameters
    ----------
    submesh :
        A manifold triangular mesh accepted by
        [mesh_to_poly][meshmash.utils.mesh_to_poly].

    Returns
    -------
    :
        Sorted array of vertex indices located on boundary edges.
    """
    # TODO currently this only works if input mesh is manifold, should relax this
    # and actually look at what edges are being broken maybe
    poly = mesh_to_poly(submesh)
    poly["index"] = np.arange(poly.n_points)
    edges = poly.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        non_manifold_edges=False,
        manifold_edges=False,
    )
    border_indices = np.array(edges["index"], dtype=int)
    return border_indices


class MeshStitcher:
    """Split a mesh into overlapping chunks and apply functions across them.

    Coordinates the full split-compute-stitch workflow:

    1. Call [split_mesh][meshmash.split.MeshStitcher.split_mesh] to partition the mesh into overlapping
       submeshes.
    2. Call [apply][meshmash.split.MeshStitcher.apply] (or [apply_on_features][meshmash.split.MeshStitcher.apply_on_features] /
       [subset_apply][meshmash.split.MeshStitcher.subset_apply]) to run a function on each submesh in
       parallel.
    3. Results are stitched back to the full mesh using
       [stitch_features][meshmash.split.MeshStitcher.stitch_features].

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    verbose :
        Verbosity level.  ``0`` / ``False`` is silent; ``>=1`` prints
        progress; ``>=2`` prints extra diagnostics.
    n_jobs :
        Number of parallel workers for [Parallel][joblib.Parallel].  ``-1``
        uses all available CPU cores.
    """

    def __init__(
        self, mesh: Mesh, verbose: Union[bool, int] = False, n_jobs: Optional[int] = -1
    ) -> None:
        self.mesh = interpret_mesh(mesh)
        self.verbose = verbose
        self.n_jobs = n_jobs

    def split_mesh(
        self,
        max_vertex_threshold: int = 20_000,
        min_vertex_threshold: int = 100,
        overlap_distance: float = 20_000,
        max_rounds: int = 100000,
        max_overlap_neighbors: Optional[int] = None,
        verify_connected: bool = True,
        method: str = "bisect",
        target_vertices: int = 10_000,
    ) -> list[Mesh]:
        """Partition the mesh and build overlapping submesh chunks.

        Calls one of the fitting functions to produce non-overlapping core
        chunks, then expands each chunk by including all vertices
        reachable within ``overlap_distance`` along mesh edges.  The
        resulting submeshes, their overlap vertex indices, and the
        per-vertex submesh mapping are stored on ``self`` for use by
        [apply][meshmash.split.MeshStitcher.apply], [apply_on_features][meshmash.split.MeshStitcher.apply_on_features], and
        [stitch_features][meshmash.split.MeshStitcher.stitch_features].

        ``method`` chooses how the mesh is cut, and nothing downstream of the
        cut changes with it: the collar, the per-chunk work and the stitching
        all read the same per-vertex partition.

        Parameters
        ----------
        max_vertex_threshold :
            Maximum vertices per non-overlapping core chunk; passed to
            the fitting function.
        min_vertex_threshold :
            Minimum connected-component size; smaller components are
            discarded.
        overlap_distance :
            Maximum geodesic edge distance used to expand each core chunk
            into its overlapping neighbourhood.
        max_rounds :
            Maximum bisection rounds; passed to [fit_mesh_split][meshmash.split.fit_mesh_split].
        max_overlap_neighbors :
            If set, limits each chunk's overlap to at most this many
            additional vertices (ranked by distance).  ``None`` keeps
            all vertices within ``overlap_distance``.
        verify_connected :
            If ``True``, assert that every overlapping submesh forms a
            single connected component.
        method :
            Which cut to use.  ``"bisect"`` is recursive spectral bisection
            ([fit_mesh_split][meshmash.split.fit_mesh_split]), the default.
            ``"geodesic"`` is geodesic Voronoi cells
            ([fit_mesh_split_geodesic][meshmash.split.fit_mesh_split_geodesic]),
            which is cheaper, reproducible, and gives connected chunks.
        target_vertices :
            Wanted vertices per chunk for ``method="geodesic"``, where it
            must be a positive integer.  Ignored by ``method="bisect"``,
            which has no size target beyond ``max_vertex_threshold``.

        Returns
        -------
        :
            List of overlapping submeshes as ``(vertices, faces)``
            tuples, one per chunk.
        """
        if method not in ("bisect", "geodesic"):
            raise ValueError(f"method must be 'bisect' or 'geodesic', got {method!r}")

        if max_vertex_threshold is None:
            max_vertex_threshold = len(self.mesh[0])
        if min_vertex_threshold is None:
            min_vertex_threshold = 0

        if self.verbose:
            currtime = time.time()
            print("Subdividing mesh...")

        if method == "geodesic":
            submesh_mapping = fit_mesh_split_geodesic(
                self.mesh,
                max_vertex_threshold=max_vertex_threshold,
                min_vertex_threshold=min_vertex_threshold,
                target_vertices=target_vertices,
                max_rounds=max_rounds,
                verbose=self.verbose,
            )
        else:
            submesh_mapping = fit_mesh_split(
                self.mesh,
                max_vertex_threshold=max_vertex_threshold,
                min_vertex_threshold=min_vertex_threshold,
                max_rounds=max_rounds,
                verbose=self.verbose,
            )

        if self.verbose >= 2:
            print(f"Subdivision took {time.time() - currtime:.3f} seconds.")

        return self.expand_split(
            submesh_mapping,
            overlap_distance=overlap_distance,
            max_overlap_neighbors=max_overlap_neighbors,
            verify_connected=verify_connected,
        )

    def expand_split(
        self,
        submesh_mapping: np.ndarray,
        overlap_distance: float = 20_000,
        max_overlap_neighbors: Optional[int] = None,
        verify_connected: bool = True,
    ) -> list[Mesh]:
        """Build the overlapping submeshes for a partition computed elsewhere.

        The second half of [split_mesh][meshmash.split.MeshStitcher.split_mesh],
        on its own: given the non-overlapping core chunks, expand each one
        along mesh edges into its overlap region and store the submeshes,
        their overlap vertex indices, and ``submesh_mapping`` on ``self``.
        [split_mesh][meshmash.split.MeshStitcher.split_mesh] is exactly
        [fit_mesh_split][meshmash.split.fit_mesh_split] followed by this.

        The split of the two matters because only the first half is
        expensive and only the first half is irreproducible: the recursive
        Fiedler bisection partitions on the sign of an eigenvector that
        converges to a tolerance, while this expansion is a deterministic
        traversal.  A caller that has stored a partition can rebuild the
        same stitcher from it as many times as it likes, and two callers
        that rebuild from the same partition agree by construction.

        Parameters
        ----------
        submesh_mapping :
            Per-vertex integer array of length ``V`` naming each vertex's
            core chunk, as returned by
            [fit_mesh_split][meshmash.split.fit_mesh_split].  ``-1`` for a
            vertex in no chunk.  The other labels must run
            ``0, 1, …, K-1``, since chunk ``i`` of the returned list is the
            chunk labelled ``i``.
        overlap_distance :
            Maximum geodesic edge distance used to expand each core chunk
            into its overlapping neighbourhood.
        max_overlap_neighbors :
            If set, limits each chunk's overlap to at most this many
            additional vertices (ranked by distance).  ``None`` keeps
            all vertices within ``overlap_distance``.
        verify_connected :
            If ``True``, assert that every overlapping submesh forms a
            single connected component.

        Returns
        -------
        :
            List of overlapping submeshes as ``(vertices, faces)``
            tuples, one per chunk.

        Raises
        ------
        ValueError
            If ``submesh_mapping`` has the wrong length, or if its labels do
            not run ``0, 1, …, K-1``.
        """
        submesh_mapping = np.asarray(submesh_mapping)
        if len(submesh_mapping) != len(self.mesh[0]):
            raise ValueError(
                f"submesh_mapping has {len(submesh_mapping)} entries but the "
                f"mesh has {len(self.mesh[0])} vertices"
            )

        self.submesh_mapping = submesh_mapping

        # The chunks come from the partition itself, not from a face-derived
        # submesh list: a cell one vertex wide owns no face whose three
        # vertices share its label, so it would drop out of such a list and
        # take the position of every chunk after it with it.
        labels = np.unique(submesh_mapping)
        labels = labels[labels >= 0]
        n_chunks = len(labels)
        if n_chunks > 0 and not np.array_equal(labels, np.arange(n_chunks)):
            raise ValueError(
                "submesh_mapping labels must run 0, 1, ..., K-1, because the "
                f"chunk at position i is the chunk labelled i; got {labels}"
            )

        adjacency = mesh_to_adjacency(self.mesh)

        self.submesh_overlap_indices = []
        self.submeshes = []

        if self.verbose:
            currtime = time.time()
            print("Finding overlapping submeshes...")
        for i in range(n_chunks):
            submesh_to_original_mapping = np.where(submesh_mapping == i)[0]
            neighbor_dists = dijkstra(
                adjacency,
                directed=False,
                indices=submesh_to_original_mapping,
                unweighted=False,
                limit=overlap_distance,
                min_only=True,
            )
            neighbor_mask = np.isfinite(neighbor_dists)

            # only keep the top max_overlap_neighbors neighbors
            if max_overlap_neighbors is not None:
                ranks = rankdata(neighbor_dists, method="min", nan_policy="omit")
                is_closest = ranks < max_overlap_neighbors + 1
                neighbor_mask = neighbor_mask & is_closest

            indices = np.arange(adjacency.shape[0])
            indices = indices[neighbor_mask | (submesh_mapping == i)]

            if max_overlap_neighbors is None:
                # TODO this is a total hack but sometimes a node can get disconnected by
                # this operation since it only keeps faces where all vertices are in the
                # subset. Could change that in the future (but then we have opposite
                # problem of getting too many nodes) or do something that selects faces is
                # maybe more natural
                new_submesh = subset_mesh_by_indices(self.mesh, indices)
                adj = mesh_to_adjacency(new_submesh)
                n_components, labels = connected_components(adj)
                uni_labels, counts = np.unique(labels, return_counts=True)
                largest = uni_labels[np.argmax(counts)]
                fixed_indices = indices[labels == largest]
                new_submesh = subset_mesh_by_indices(self.mesh, fixed_indices)
            else:
                # new behavior
                new_submesh, fixed_indices = rough_subset_mesh_by_indices(
                    self.mesh, indices
                )

            # check if all submeshes are one connected component
            if verify_connected:
                adj = mesh_to_adjacency(new_submesh)
                n_components, labels = connected_components(adj)
                assert n_components == 1

            self.submesh_overlap_indices.append(fixed_indices)
            self.submeshes.append(new_submesh)

        # self.submeshes = [
        #     subset_mesh_by_indices(self.mesh, indices)
        #     for indices in self.submesh_overlap_indices
        # ]
        # for submesh in self.submeshes:
        #     adj = mesh_to_adjacency(submesh)
        #     assert connected_components(adj)[0] == 1
        if self.verbose >= 2:
            print(f"Overlap detection took {time.time() - currtime:.3f} seconds.")

        return self.submeshes

    def stitch_features(
        self,
        features_by_submesh: list[np.ndarray],
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """Combine per-submesh feature arrays into a single full-mesh array.

        Each vertex in the overlap region is covered by multiple submeshes.
        This method assigns each vertex the value computed by the submesh
        that "owns" it (i.e. ``submesh_mapping[vertex] == submesh_index``).

        Parameters
        ----------
        features_by_submesh :
            List of 2-D feature arrays, one per submesh, aligned to the
            submesh vertex ordering.  ``None`` entries are skipped.
        fill_value :
            Value used to pre-fill the output array before writing submesh
            results.  Vertices in chunks where the function returned
            ``None`` retain this value.

        Returns
        -------
        :
            Full-mesh feature array of shape ``(V, n_features)``.
        """
        valid_features = [feat for feat in features_by_submesh if feat is not None]
        if len(valid_features) == 0:
            raise ValueError("All features are None")
        first_feature = valid_features[0]
        all_clean_features = np.full(
            (len(self.mesh[0]), first_feature.shape[1]),
            fill_value,
            dtype=first_feature.dtype,
        )

        for i, features in enumerate(features_by_submesh):
            if features is not None:
                indices_in_original = self.submesh_overlap_indices[i]
                keep_mask = self.submesh_mapping[indices_in_original] == i
                keep_features = features[keep_mask]
                index = np.where(self.submesh_mapping == i)
                all_clean_features[index] = keep_features

        # if add_label_column:
        #     mapping = self.submesh_mapping
        #     valids = mapping[mapping != -1]
        #     submesh_labels, submesh_label_counts = np.unique(
        #         valids, return_counts=True
        #     )
        #     submesh_indicator = [
        #         np.full(count, label)
        #         for label, count in zip(submesh_labels, submesh_label_counts)
        #     ]
        #     submesh_indicator = np.concatenate(submesh_indicator, axis=0)
        #     all_clean_features = np.concatenate(
        #         [all_clean_features, submesh_indicator[:, None]], axis=1
        #     )

        return all_clean_features

    def apply(
        self,
        func: Callable[..., Any],
        *args,
        fill_value: float = np.nan,
        stitch: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, list]:
        """Apply a function to each submesh and (optionally) stitch results.

        ``func`` must have signature ``func(submesh, *args, **kwargs)``
        and return a 2-D array aligned to the submesh vertices, or
        ``None`` to indicate failure.

        Parameters
        ----------
        func :
            Callable to apply to each submesh.
        *args :
            Positional arguments forwarded to ``func`` after ``submesh``.
        fill_value :
            Fill value for uncomputed vertices; see [stitch_features][meshmash.split.MeshStitcher.stitch_features].
        stitch :
            If ``True`` (default), stitch results into a full-mesh array
            via [stitch_features][meshmash.split.MeshStitcher.stitch_features].  If ``False``, return the raw
            list of per-submesh results.
        **kwargs :
            Keyword arguments forwarded to ``func``.

        Returns
        -------
        :
            Stitched full-mesh feature array of shape ``(V, n_features)``
            when ``stitch=True``, or the raw list of per-submesh results
            when ``stitch=False``.
        """
        func_name = func.__name__
        submeshes = self.submeshes
        if self.n_jobs == 1:
            results_by_submesh = []
            for submesh in tqdm(
                submeshes,
                desc=f"Applying function {func_name}",
                disable=self.verbose < 1,
            ):
                results_by_submesh.append(func(submesh, *args, **kwargs))
        else:
            with tqdm_joblib(
                desc=f"Applying function {func_name}",
                total=len(submeshes),
                disable=self.verbose < 1,
            ):
                results_by_submesh = Parallel(n_jobs=self.n_jobs)(
                    delayed(func)(submesh, *args, **kwargs) for submesh in submeshes
                )
        if stitch:
            out = self.stitch_features(results_by_submesh, fill_value=fill_value)
        else:
            out = results_by_submesh
        return out

    def subset_apply(
        self,
        func: Callable[..., Any],
        indices: np.ndarray,
        *args,
        reindex: bool = False,
        fill_value: float = np.nan,
        **kwargs,
    ) -> np.ndarray:
        """Apply a function only to submeshes that contain ``indices``.

        Useful when features are only needed for a subset of vertices;
        submeshes that do not overlap with ``indices`` are skipped.

        Parameters
        ----------
        func :
            Callable with signature ``func(submesh, *args, **kwargs)``.
        indices :
            Vertex indices of interest.  Only submeshes that contain at
            least one of these vertices are processed.
        *args :
            Positional arguments forwarded to ``func``.
        reindex :
            If ``True``, return only the rows of the stitched result
            corresponding to ``indices``.  If ``False`` (default), return
            the full-mesh array.
        fill_value :
            Fill value for vertices whose submesh was not processed.
        **kwargs :
            Keyword arguments forwarded to ``func``.

        Returns
        -------
        :
            Full-mesh feature array of shape ``(V, n_features)`` or, when
            ``reindex=True``, shape ``(len(indices), n_features)``.
        """
        func_name = func.__name__
        index_submesh_mappings = self.submesh_mapping[indices]
        relevant_submesh_indices = np.unique(index_submesh_mappings)
        relevant_submesh_indices = relevant_submesh_indices[
            relevant_submesh_indices != -1
        ]
        submeshes = self.submeshes
        relevant_submeshes = [submeshes[i] for i in relevant_submesh_indices]
        if self.n_jobs == 1:
            results_by_relevant_submesh = []
            for submesh in tqdm(
                relevant_submeshes,
                desc=f"Applying function {func_name}",
                disable=self.verbose < 1,
            ):
                results_by_relevant_submesh.append(func(submesh, *args, **kwargs))
        else:
            with tqdm_joblib(
                desc=f"Applying function {func_name}",
                total=len(relevant_submeshes),
                disable=self.verbose < 1,
            ):
                results_by_relevant_submesh = Parallel(n_jobs=self.n_jobs)(
                    delayed(func)(submesh, *args, **kwargs)
                    for submesh in relevant_submeshes
                )

        results_by_submesh = []
        counter = 0
        for index in range(len(submeshes)):
            if index in relevant_submesh_indices:
                results_by_submesh.append(results_by_relevant_submesh[counter])
                counter += 1
            else:
                results_by_submesh.append(None)

        # TODO this is lazy bc I didn't want to rewrite the stitch_features function
        stitched_features = self.stitch_features(
            results_by_submesh, fill_value=fill_value
        )
        if reindex:
            return stitched_features[indices]
        else:
            return stitched_features

    def apply_on_features(
        self,
        func: Callable[..., Any],
        X: np.ndarray,
        *args,
        fill_value: float = np.nan,
        stitch: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, list]:
        """Apply a function that takes both a submesh and a feature slice.

        Like [apply][meshmash.split.MeshStitcher.apply], but also passes the slice of ``X`` corresponding
        to each submesh as the second argument to ``func``.  The expected
        signature is ``func(submesh, submesh_features, *args, **kwargs)``.

        Parameters
        ----------
        func :
            Callable with signature
            ``func(submesh, submesh_features, *args, **kwargs)``.
        X :
            Full-mesh feature array of shape ``(V, F)``.  Each submesh
            receives the rows indexed by its overlap vertex indices.
        *args :
            Additional positional arguments forwarded to ``func``.
        fill_value :
            Fill value for uncomputed vertices.
        stitch :
            If ``True`` (default), stitch results into a full-mesh array
            via [stitch_features][meshmash.split.MeshStitcher.stitch_features].  If ``False``, return the raw
            list of per-submesh results — which is what a caller returning
            more than an array per submesh needs.
        **kwargs :
            Keyword arguments forwarded to ``func``.

        Returns
        -------
        :
            Stitched full-mesh result array of shape ``(V, n_out_features)``
            when ``stitch=True``, or the raw list of per-submesh results
            when ``stitch=False``.
        """
        func_name = func.__name__
        submeshes = self.submeshes
        if self.n_jobs == 1:
            results_by_submesh = []
            for i, submesh in enumerate(
                tqdm(
                    submeshes,
                    desc=f"Applying function {func_name}",
                    disable=self.verbose < 1,
                )
            ):
                submesh_features = X[self.submesh_overlap_indices[i]]
                results_by_submesh.append(
                    func(submesh, submesh_features, *args, **kwargs)
                )
        else:
            with tqdm_joblib(
                desc=f"Applying function {func_name}",
                total=len(submeshes),
                disable=self.verbose < 1,
            ):
                results_by_submesh = Parallel(n_jobs=self.n_jobs)(
                    delayed(func)(
                        submesh, X[self.submesh_overlap_indices[i]], *args, **kwargs
                    )
                    for i, submesh in enumerate(submeshes)
                )

        if not stitch:
            return results_by_submesh

        out_features = self.stitch_features(results_by_submesh, fill_value=fill_value)

        return out_features
