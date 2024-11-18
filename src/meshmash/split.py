import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components, dijkstra, laplacian
from scipy.sparse.linalg import eigsh
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

from .types import Mesh, interpret_mesh
from .utils import mesh_to_adjacency, mesh_to_poly


def subset_mesh_by_indices(mesh: Mesh, indices: np.ndarray) -> Mesh:
    vertices, faces = interpret_mesh(mesh)
    new_vertices = vertices[indices]
    index_mapping = dict(zip(indices, np.arange(len(indices))))
    # use numpy to get faces for which all indices are in the subset
    face_mask = np.all(np.isin(faces, indices), axis=1)
    new_faces = np.vectorize(index_mapping.get)(faces[face_mask])
    return new_vertices, new_faces


def graph_laplacian_split(adj: csr_array):
    # TODO normed didn't seem to make much of a difference here; perhaps just because
    # degrees are fairly homogeneous?
    lap, degrees = laplacian(adj, normed=False, symmetrized=True, return_diag=True)

    # NOTE: tried this as initialization, but it also didn't seem to make a difference
    # maybe overhead is all in the LU decomposition?
    # n = adj.shape[0]
    # v0 = np.full(n, 1 / np.sqrt(n))
    eigenvalues, eigenvectors = eigsh(
        lap,
        k=2,
        sigma=-1e-10,
    )
    indices1 = np.nonzero(eigenvectors[:, 1] >= 0)[0]
    indices2 = np.nonzero(eigenvectors[:, 1] < 0)[0]
    return indices1, indices2


def bisect_adjacency(adj: csr_array):
    # get the split indices
    indices1, indices2 = graph_laplacian_split(adj)

    # get the sub-adjacencies
    sub_adj1 = adj[indices1][:, indices1]
    sub_adj2 = adj[indices2][:, indices2]

    # make sure we didn't disconnect any nodes
    degrees1 = np.sum(sub_adj1, axis=1) + np.sum(sub_adj1, axis=0)
    degrees2 = np.sum(sub_adj2, axis=1) + np.sum(sub_adj2, axis=0)
    if np.any(degrees1 == 0):
        raise RuntimeError("Some nodes were disconnected in the split.")
    if np.any(degrees2 == 0):
        raise RuntimeError("Some nodes were disconnected in the split.")

    sub_adjs = (sub_adj1, sub_adj2)
    submesh_indices = (indices1, indices2)

    return sub_adjs, submesh_indices


def fit_mesh_split(mesh: Mesh, vertex_threshold=20_000, max_rounds=1000, verbose=False):
    mesh = interpret_mesh(mesh)
    n_vertices = mesh[0].shape[0]
    mesh_indices = np.arange(n_vertices)
    whole_adj = mesh_to_adjacency(mesh)
    n_components, component_labels = connected_components(whole_adj)

    adj_queue = []
    for component_id in tqdm(range(n_components)):
        component_mask = component_labels == component_id
        count = component_mask.sum()
        if count > 100:
            component_indices = mesh_indices[component_mask]
            component_adj = whole_adj[component_indices][:, component_indices]
            assert connected_components(component_adj)[0] == 1
            adj_queue.append((component_adj, component_indices))

    # for component_id in tqdm(range(whole_mesh_poly["RegionId"].max() + 1)):
    #     currtime = time.time()
    #     component_mask = whole_mesh_poly.point_data["RegionId"] == component_id
    #     count = component_mask.sum()
    #     if count > 50:
    #         component_indices = mesh_indices[component_mask]
    #         # TODO this operation seems much slower than it needs to be
    #         # component_mesh = whole_mesh_poly.extract_points(
    #         #     component_indices, adjacent_cells=False
    #         # ).extract_surface().triangulate()
    #         # extract_time += time.time() - currtime
    #         # # print(type(component_poly))
    #         # # component_mesh = subset_mesh_by_indices(mesh, component_indices)
    #         # currtime = time.time()
    #         # component_adj = mesh_to_adjacency(component_mesh)
    #         component_adj = whole_adj[component_indices][:, component_indices]
    #         assert connected_components(component_adj)[0] == 1
    #         adj_time += time.time() - currtime
    #         adj_queue.append((component_adj, component_indices))

    submesh_mapping = np.full(n_vertices, -1, dtype=int)
    indices_by_submesh = []

    n_finished = 0
    rounds = 0

    while len(adj_queue) > 0 and rounds < max_rounds:
        if verbose:
            # print("Meshes in queue:", [m[0].shape[0] for m in adj_queue])
            print("Meshes in queue:", len(adj_queue))
        current_adj, current_indices = adj_queue.pop(0)

        if current_adj.shape[0] <= vertex_threshold:
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
            if sub_adj.shape[0] > vertex_threshold:
                adj_queue.append((sub_adj, indices))
            else:
                assert connected_components(sub_adj)[0] == 1
                # finished_meshes.append((sub_adj, indices))
                submesh_mapping[indices] = n_finished
                indices_by_submesh.append(indices)
                n_finished += 1
        rounds += 1
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
        self, vertex_threshold=20_000, overlap_distance=20_000, max_rounds=1000
    ):
        if self.verbose:
            currtime = time.time()
            print("Subdividing mesh...")

        submesh_mapping = fit_mesh_split(
            self.mesh,
            vertex_threshold=vertex_threshold,
            max_rounds=max_rounds,
            verbose=self.verbose,
        )

        self.submesh_mapping = submesh_mapping
        temp_submeshes = apply_mesh_split(self.mesh, submesh_mapping)

        # check if all submeshes are one connected component
        for submesh in temp_submeshes:
            poly = mesh_to_poly(submesh)
            assert poly.n_points == poly.extract_largest().n_points

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
            indices = np.arange(adjacency.shape[0])
            indices = indices[neighbor_mask | (submesh_mapping == i)]
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

            # check if all submeshes are one connected component
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
        if self.verbose:
            print(f"Overlap detection took {time.time() - currtime:.3f} seconds.")

        return self.submeshes

    def stitch_features(
        self,
        features_by_submesh: list[np.ndarray],
        fill_value=np.nan,
    ) -> np.ndarray:
        all_clean_features = np.full(
            (len(self.mesh[0]), features_by_submesh[0].shape[1]),
            fill_value,
            dtype=float,
        )
        for i, features in enumerate(features_by_submesh):
            indices_in_original = self.submesh_overlap_indices[i]
            keep_mask = self.submesh_mapping[indices_in_original] == i
            keep_features = features[keep_mask]
            index = np.where(self.submesh_mapping == i)
            all_clean_features[index] = keep_features
        return all_clean_features

    def apply(
        self,
        func,
        *args,
        fill_value=np.nan,
        **kwargs,
    ):
        submeshes = self.submeshes
        if self.n_jobs == 1:
            results_by_submesh = []
            for submesh in tqdm(
                submeshes,
                desc="Applying function over submeshes",
                disable=self.verbose < 1,
            ):
                results_by_submesh.append(func(submesh, *args, **kwargs))
        else:
            with tqdm_joblib(
                desc="Applying function over submeshes",
                total=len(submeshes),
                disable=self.verbose < 1,
            ):
                results_by_submesh = Parallel(n_jobs=self.n_jobs)(
                    delayed(func)(submesh, *args, **kwargs) for submesh in submeshes
                )

        return self.stitch_features(results_by_submesh, fill_value=fill_value)