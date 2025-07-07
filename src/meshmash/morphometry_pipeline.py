import fastremap
import numpy as np
import pandas as pd
from gpytoolbox import fast_winding_number
from point_cloud_utils import closest_points_on_mesh
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from .decompose import decompose_mesh
from .laplacian import compute_vertex_areas
from .utils import get_label_components, mesh_to_adjacency


def component_morphometry_pipeline(
    mesh,
    labels,
    select_label,
    post_synapse_mappings=None,
    split_laplacian="graph",
    split_threshold=1.25,
    split_min_size=10,
    verbose=False,
):
    # TODO would like to generalize this

    components = get_label_components(mesh, labels)

    vertices = mesh[0].copy()
    faces = mesh[1].copy()

    # only care about some components to measure, e.g. spines, boutons, etc.
    select_component_ids = np.unique(components[labels == select_label])

    # this sets up a faster way of getting subfaces that belong to a component
    face_components = np.stack([components[face] for face in faces.T]).T
    select_component_mask = np.isin(face_components[:, 0], select_component_ids)
    # TODO this makes intuitive sense to have but something breaks when I uncomment it
    same_component_mask = (face_components[:, 0] == face_components[:, 1]) & (
        face_components[:, 1] == face_components[:, 2]
    )
    select_component_mask = same_component_mask & select_component_mask
    select_component_faces = faces[select_component_mask]
    select_component_faces_labels = face_components[select_component_mask][:, 0].copy()

    corrected_components = components.copy()

    areas = compute_vertex_areas(mesh, robust=False)

    indices = np.arange(len(mesh[0]))
    points_per_um3 = 25_000

    rows = []
    component_queue = list(np.unique(select_component_faces_labels))
    next_id = np.max(components) + 1
    with tqdm(total=len(component_queue), disable=not verbose) as pbar:
        while len(component_queue) > 0:
            select_component_id = component_queue.pop(0)
            row_data = {}
            row_data["select_component_id"] = select_component_id

            select_component_face_indices = np.where(
                select_component_faces_labels == select_component_id
            )[0]
            subfaces = select_component_faces[select_component_face_indices]
            used_vertices = np.unique(subfaces)

            if len(used_vertices) < 10:
                continue

            index_map = dict(zip(used_vertices, np.arange(len(used_vertices))))
            reverse_index_map = {v: k for k, v in index_map.items()}
            subfaces = fastremap.remap(subfaces, index_map)
            subvertices = vertices[used_vertices]

            submesh = (subvertices, subfaces)

            if split_laplacian == "graph":
                subadj = mesh_to_adjacency(submesh)
                L = laplacian(subadj, normed=False, symmetrized=True)
                evals, evecs = eigsh(L, k=2, sigma=-1e-10, return_eigenvectors=True)
                indices = np.argsort(evals)
                evals = evals[indices]
                evecs = evecs[:, indices]
                fiedler_eval = evals[1]
                fiedler_evec = evecs[:, 1]
            elif split_laplacian == "mesh":
                evals, evecs = decompose_mesh(
                    submesh,
                    n_components=2,
                    robust=True,
                )
                fiedler_eval = evals[1]
                fiedler_evec = evecs[:, 1]
            else:
                fiedler_eval = 0

            if (split_threshold is not None) and (fiedler_eval < split_threshold):
                split_indices_1 = np.where(fiedler_evec > 0)[0]
                split_indices_2 = np.where(fiedler_evec <= 0)[0]
                if (
                    len(split_indices_1) >= split_min_size
                    and len(split_indices_2) >= split_min_size
                ):
                    split_indices_1 = fastremap.remap(
                        split_indices_1, reverse_index_map
                    )
                    split_indices_2 = fastremap.remap(
                        split_indices_2, reverse_index_map
                    )

                    id1 = next_id
                    mapping = {}
                    for vertex_index in split_indices_1:
                        mapping[vertex_index] = next_id
                    corrected_components[split_indices_1] = next_id
                    next_id += 1

                    id2 = next_id
                    for vertex_index in split_indices_2:
                        mapping[vertex_index] = next_id
                    corrected_components[split_indices_2] = next_id
                    next_id += 1
                    original_subfaces = select_component_faces[
                        select_component_face_indices
                    ]
                    remapped_subfaces_component_labels = fastremap.remap(
                        original_subfaces, mapping
                    )
                    subfaces_is_split1 = (
                        remapped_subfaces_component_labels == id1
                    ).all(axis=1)
                    subfaces_is_split2 = (
                        remapped_subfaces_component_labels == id2
                    ).all(axis=1)

                    select_component_faces_labels[
                        select_component_face_indices[subfaces_is_split1]
                    ] = id1
                    select_component_faces_labels[
                        select_component_face_indices[subfaces_is_split2]
                    ] = id2

                    component_queue.append(id1)
                    component_queue.append(id2)

                    pbar.update(-1)
                    continue

            row_data[f"{split_laplacian}_fiedler_eval"] = np.float32(fiedler_eval)

            bounds = np.array([submesh[0].min(axis=0), submesh[0].max(axis=0)])
            bound_volume_um3 = np.prod(bounds[1] - bounds[0]) * 1e-9

            n_points = max(int(bound_volume_um3 * points_per_um3), 1000)
            # TODO make this just a grid
            sample_points = np.random.uniform(bounds[0], bounds[1], (n_points, 3))
            winding_numbers = fast_winding_number(sample_points, submesh[0], submesh[1])

            inside_mask = winding_numbers > 0.5
            n_interior_samples = inside_mask.sum()
            row_data["n_interior_samples"] = np.int32(n_interior_samples)

            p_in_vol = (inside_mask).sum() / len(sample_points)
            vol_estimate = p_in_vol * bound_volume_um3 * 1e9  # convert to nm^3
            row_data["size_nm3"] = np.float32(vol_estimate)

            area = areas[used_vertices].sum()
            row_data["area_nm2"] = np.float32(area)

            sphericity = (np.pi) ** (1 / 3) * (6 * vol_estimate) ** (2 / 3) / area
            row_data["sphericity"] = np.float32(sphericity)

            inside_points = sample_points[inside_mask]
            if inside_points.shape[0] > 5:
                subsample_indices = np.random.choice(
                    inside_points.shape[0],
                    size=min(250, len(inside_points)),
                    replace=False,
                )
                subsample_points = inside_points[subsample_indices]
                pdist = pairwise_distances(subsample_points, metric="euclidean")
                medioid = subsample_points[np.argmin(pdist.sum(axis=1))]
                row_data["x"] = np.float32(medioid[0])
                row_data["y"] = np.float32(medioid[1])
                row_data["z"] = np.float32(medioid[2])

                subsample_indices = np.random.choice(
                    inside_points.shape[0],
                    size=min(1000, len(inside_points)),
                    replace=False,
                )
                subsample_points = inside_points[subsample_indices]
                pca = PCA(n_components=3)
                pca.fit(subsample_points)
                row_data["pca_val_1"] = np.float32(pca.singular_values_[0])
                row_data["pca_val_2"] = np.float32(pca.singular_values_[1])
                row_data["pca_val_3"] = np.float32(pca.singular_values_[2])

                dists, _, _ = closest_points_on_mesh(
                    subsample_points.astype(subvertices.dtype), subvertices, subfaces
                )
                row_data["max_dt_nm"] = np.float32(dists.max())
                row_data["mean_dt_nm"] = np.float32(dists.mean())

            rows.append(row_data)
            pbar.update(1)

    results = (
        pd.DataFrame(rows)
        .set_index("select_component_id")
        .rename_axis(index="component_id")
    )
    results = results.dropna(axis=0, how="any")

    corrected_components[~np.isin(corrected_components, results.index)] = -1
    corrected_components = corrected_components.astype(np.int32)

    if post_synapse_mappings is not None:
        post_synapse_components = corrected_components[post_synapse_mappings]
        results["n_post_synapses"] = np.zeros(len(results), dtype=np.uint16)
        for component in post_synapse_components:
            if component != -1:
                results.loc[component, "n_post_synapses"] += 1

    return results, corrected_components
