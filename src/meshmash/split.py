import logging
import time
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_array
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


def graph_laplacian_split(adj: csr_array, dtype=np.float32):
    # TODO didn't understand why this took longer when float32 for some meshes
    # probably some issue with tolerance/sigma?
    # TODO normed didn't seem to make much of a difference here; perhaps just because
    # degrees are fairly homogeneous?
    lap = laplacian(adj, normed=False, symmetrized=True, return_diag=False, dtype=dtype)
    if dtype == np.float32 or dtype == "float32":
        eigen_tol = 1e-7
    elif dtype == np.float64 or dtype == "float64":
        eigen_tol = 1e-10
    n = adj.shape[0]

    # TODO cannot figure out why this isn't deterministic
    # or if the random errors are from some other part of the pipeline
    eigenvalues, eigenvectors = eigsh(
        lap,
        k=2,
        sigma=-1e-10,
        v0=np.full(n, 1 / np.sqrt(n), dtype=dtype),
        tol=eigen_tol,
        maxiter=20,
        ncv=20, # TODO revisit this sensitivity to NCV for speed
    )

    index = np.argmax(eigenvalues)
    eigenvector = eigenvectors[:, index]
    eigenvector *= np.sign(eigenvector[0]) * 1
    indices1 = np.nonzero(eigenvector >= 0)[0]
    indices2 = np.nonzero(eigenvector < 0)[0]

    return indices1, indices2


def bisect_adjacency(adj: csr_array, n_retries: int = 7, check=True):
    if n_retries == 0:
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


def fit_mesh_split(
    mesh: Union[Mesh, np.ndarray, csr_array],
    max_vertex_threshold=20_000,
    min_vertex_threshold=100,
    max_rounds=100000,
    verbose=False,
):
    if isinstance(mesh, (csr_array, np.ndarray)):
        whole_adj = mesh
    else:
        mesh = interpret_mesh(mesh)
        whole_adj = mesh_to_adjacency(mesh)

    n_vertices = whole_adj.shape[0]
    mesh_indices = np.arange(n_vertices)

    # first, append all the connected components that are large enough to the queue
    n_components, component_labels = connected_components(whole_adj)

    adj_queue = []
    for component_id in range(n_components):
        component_mask = component_labels == component_id
        count = component_mask.sum()
        if count >= min_vertex_threshold:
            component_indices = mesh_indices[component_mask]
            component_adj = whole_adj[component_indices][:, component_indices]
            adj_queue.append((component_adj, component_indices))

    submesh_mapping = np.full(n_vertices, -1, dtype=int)
    indices_by_submesh = []

    n_finished = 0
    rounds = 0

    while len(adj_queue) > 0 and rounds < max_rounds:
        if verbose and rounds % 50 == 0:
            print("Meshes in queue:", len(adj_queue))
        current_adj, current_indices = adj_queue.pop(0)

        # if this submesh is small enough, add it to the finished list
        # this can happen if ccs are already small
        if current_adj.shape[0] <= max_vertex_threshold:
            sub_adjs, submesh_indices_to_main = [current_adj], [current_indices]
        else:  # otherwise, split
            sub_adjs, submesh_indices = bisect_adjacency(current_adj)
            submesh_colors = np.zeros(n_vertices, dtype=float)
            submesh_colors[submesh_indices[0]] = 0
            submesh_colors[submesh_indices[1]] = 1

            # adjust indices to be in terms of the main mesh
            submesh_indices_to_main = [
                current_indices[indices] for indices in submesh_indices
            ]

        for sub_adj, indices in zip(sub_adjs, submesh_indices_to_main):
            if sub_adj.shape[0] > max_vertex_threshold:
                adj_queue.append((sub_adj, indices))
            else:
                # TODO maybe add ensure_connected as a flag?
                # assert connected_components(sub_adj)[0] == 1
                # finished_meshes.append((sub_adj, indices))
                submesh_mapping[indices] = n_finished
                indices_by_submesh.append(indices)
                n_finished += 1
        rounds += 1

    # remap so the first submesh is largest
    valid_submesh_mapping = submesh_mapping[submesh_mapping != -1]
    labels, counts = np.unique(valid_submesh_mapping, return_counts=True)
    reorder = np.argsort(-counts)
    new_labels = np.arange(labels.max() + 1)
    old_to_new = dict(zip(labels[reorder], new_labels))
    old_to_new[-1] = -1
    submesh_mapping = np.vectorize(old_to_new.get)(submesh_mapping)

    return submesh_mapping


# def laplacian_split(L: csr_array, M):
#     # TODO normed didn't seem to make much of a difference here; perhaps just because
#     # degrees are fairly homogeneous?
#     # lap, degrees = laplacian(adj, normed=False, symmetrized=True, return_diag=True)

#     # NOTE: tried this as initialization, but it also didn't seem to make a difference
#     # maybe overhead is all in the LU decomposition?
#     # n = adj.shape[0]
#     # v0 = np.full(n, 1 / np.sqrt(n))
#     eigenvalues, eigenvectors = eigsh(
#         L,

#         k=2,
#         sigma=-1e-10,
#     )
#     indices1 = np.nonzero(eigenvectors[:, 1] >= 0)[0]
#     indices2 = np.nonzero(eigenvectors[:, 1] < 0)[0]
#     return indices1, indices2

from scipy.sparse import diags_array


def subset_diags(matrix, indices):
    return diags_array(matrix.diagonal()[indices], shape=(len(indices), len(indices)))


def bisect_laplacian(L, M):
    # get the split indices
    # indices1, indices2 = graph_laplacian_split(adj)

    _, eigenvectors = decompose_laplacian(L, M, n_components=2)
    indices1 = np.nonzero(eigenvectors[:, 1] >= 0)[0]
    indices2 = np.nonzero(eigenvectors[:, 1] < 0)[0]

    # get the sub-adjacencies
    sub_adj1 = L[indices1][:, indices1]
    sub_adj2 = L[indices2][:, indices2]

    # make sure we didn't disconnect any nodes
    # degrees1 = np.sum(sub_adj1, axis=1) + np.sum(sub_adj1, axis=0)
    # degrees2 = np.sum(sub_adj2, axis=1) + np.sum(sub_adj2, axis=0)
    # if np.any(degrees1 == 0):
    #     raise RuntimeError("Some nodes were disconnected in the split.")
    # if np.any(degrees2 == 0):
    #     raise RuntimeError("Some nodes were disconnected in the split.")

    sub_laps = (
        (sub_adj1, subset_diags(M, indices1)),
        (sub_adj2, subset_diags(M, indices2)),
    )

    submesh_indices = (indices1, indices2)

    return sub_laps, submesh_indices


def fit_mesh_split_lap(
    mesh: Union[Mesh, np.ndarray, csr_array],
    max_vertex_threshold=20_000,
    min_vertex_threshold=100,
    max_rounds=100000,
    robust=True,
    mollify_factor=1e-5,
    verbose=False,
):
    if isinstance(mesh, (csr_array, np.ndarray)):
        whole_adj = mesh
    else:
        mesh = interpret_mesh(mesh)
        whole_adj = mesh_to_adjacency(mesh)

    n_vertices = whole_adj.shape[0]
    mesh_indices = np.arange(n_vertices)

    # first, append all the connected components that are large enough to the queue
    n_components, component_labels = connected_components(whole_adj)

    adj_queue = []
    for component_id in range(n_components):
        component_mask = component_labels == component_id
        count = component_mask.sum()
        if count >= min_vertex_threshold:
            component_indices = mesh_indices[component_mask]
            # component_adj = whole_adj[component_indices][:, component_indices]
            submesh = subset_mesh_by_indices(mesh, component_indices)
            L, M = cotangent_laplacian(
                submesh, robust=robust, mollify_factor=mollify_factor
            )
            adj_queue.append(((L, M), component_indices))

    submesh_mapping = np.full(n_vertices, -1, dtype=int)
    indices_by_submesh = []

    n_finished = 0
    rounds = 0

    while len(adj_queue) > 0 and rounds < max_rounds:
        if verbose:
            print("Meshes in queue:", len(adj_queue))
        current_adj, current_indices = adj_queue.pop(0)

        # if this submesh is small enough, add it to the finished list
        # this can happen if ccs are already small
        if current_adj[0].shape[0] <= max_vertex_threshold:
            sub_adjs, submesh_indices_to_main = [current_adj], [current_indices]
        else:  # otherwise, split
            sub_adjs, submesh_indices = bisect_laplacian(*current_adj)
            submesh_colors = np.zeros(n_vertices, dtype=float)
            submesh_colors[submesh_indices[0]] = 0
            submesh_colors[submesh_indices[1]] = 1

            # adjust indices to be in terms of the main mesh
            submesh_indices_to_main = [
                current_indices[indices] for indices in submesh_indices
            ]

        for sub_adj, indices in zip(sub_adjs, submesh_indices_to_main):
            if sub_adj[0].shape[0] > max_vertex_threshold:
                adj_queue.append((sub_adj, indices))
            else:
                # TODO maybe add ensure_connected as a flag?
                # assert connected_components(sub_adj)[0] == 1
                # finished_meshes.append((sub_adj, indices))
                submesh_mapping[indices] = n_finished
                indices_by_submesh.append(indices)
                n_finished += 1
        rounds += 1

    # remap so the first submesh is largest
    valid_submesh_mapping = submesh_mapping[submesh_mapping != -1]
    labels, counts = np.unique(valid_submesh_mapping, return_counts=True)
    reorder = np.argsort(-counts)
    new_labels = np.arange(labels.max() + 1)
    old_to_new = dict(zip(labels[reorder], new_labels))
    old_to_new[-1] = -1
    submesh_mapping = np.vectorize(old_to_new.get)(submesh_mapping)

    return submesh_mapping


def apply_mesh_split(mesh: Mesh, split_mapping: np.ndarray) -> list[Mesh]:
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


def get_submesh_borders(submesh):
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


def fit_overlapping_mesh_split(
    mesh, overlap_distance=20_000, vertex_threshold=20_000, max_rounds=1_000
):
    mesh = interpret_mesh(mesh)
    submesh_mapping = fit_mesh_split(
        mesh, vertex_threshold=vertex_threshold, max_rounds=max_rounds
    )
    submeshes = apply_mesh_split(mesh, submesh_mapping)
    for submesh in submeshes:
        poly = mesh_to_poly(submesh)
        assert poly.n_points == poly.extract_largest().n_points

    adjacency = mesh_to_adjacency(mesh)
    new_indices_by_submesh = []

    for i, submesh in enumerate(submeshes):
        # border_indices = get_submesh_borders(submesh)
        submesh_to_original_mapping = np.where(submesh_mapping == i)[0]
        # border_indices = submesh_to_original_mapping[border_indices]
        neighbor_dists = dijkstra(
            adjacency,
            directed=False,
            indices=submesh_to_original_mapping,
            unweighted=False,
            limit=overlap_distance,
            min_only=True,
        )
        neighbor_mask = np.isfinite(neighbor_dists)
        indices = np.arange(adjacency.shape[0])
        indices = indices[neighbor_mask | (submesh_mapping == i)]
        new_indices_by_submesh.append(indices)
        assert connected_components(adjacency[indices][:, indices])[0] == 1
    return new_indices_by_submesh


# def apply_overlapping_mesh_split(mesh, indices_by_submesh):
#     poly = mesh_to_poly(mesh)
#     submeshes = []
#     for indices in indices_by_submesh:
#         sub_poly = (
#             poly.extract_points(indices, adjacent_cells=False)
#             .triangulate()
#             .extract_surface()
#         )
#         submeshes.append(poly_to_mesh(sub_poly))
#     return submeshes


class MeshStitcher:
    def __init__(self, mesh: Mesh, verbose=False, n_jobs=-1):
        self.mesh = interpret_mesh(mesh)
        self.verbose = verbose
        self.n_jobs = n_jobs

    def split_mesh(
        self,
        max_vertex_threshold=20_000,
        min_vertex_threshold=100,
        overlap_distance=20_000,
        max_rounds=100000,
        max_overlap_neighbors=None,
        verify_connected=True,
    ):
        if max_vertex_threshold is None:
            max_vertex_threshold = len(self.mesh[0])
        if min_vertex_threshold is None:
            min_vertex_threshold = 0

        if self.verbose:
            currtime = time.time()
            print("Subdividing mesh...")

        submesh_mapping = fit_mesh_split(
            self.mesh,
            max_vertex_threshold=max_vertex_threshold,
            min_vertex_threshold=min_vertex_threshold,
            max_rounds=max_rounds,
            verbose=self.verbose,
        )

        self.submesh_mapping = submesh_mapping
        temp_submeshes = apply_mesh_split(self.mesh, submesh_mapping)

        # # check if all submeshes are one connected component
        # for submesh in temp_submeshes:
        #     poly = mesh_to_poly(submesh)
        #     assert poly.n_points == poly.extract_largest().n_points

        if self.verbose >= 2:
            print(f"Subdivision took {time.time() - currtime:.3f} seconds.")

        adjacency = mesh_to_adjacency(self.mesh)

        self.submesh_overlap_indices = []
        self.submeshes = []

        if self.verbose:
            currtime = time.time()
            print("Finding overlapping submeshes...")
        for i, submesh in enumerate(temp_submeshes):
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
        fill_value=np.nan,
        # add_label_column=False,
    ) -> np.ndarray:
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
        func,
        *args,
        fill_value=np.nan,
        stitch=True,
        **kwargs,
    ):
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
        func,
        indices,
        *args,
        reindex=False,
        fill_value=np.nan,
        **kwargs,
    ):
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

    def apply_on_features(self, func, X, *args, fill_value=np.nan, **kwargs):
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

        out_features = self.stitch_features(results_by_submesh, fill_value=fill_value)

        return out_features
