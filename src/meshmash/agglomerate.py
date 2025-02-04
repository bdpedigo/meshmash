import numpy as np
import pandas as pd
from sklearn.cluster import ward_tree

# TODO dangerous to import private function here
from sklearn.cluster._agglomerative import _hc_cut

from .split import MeshStitcher
from .utils import mesh_to_adjacency


def multicut_ward(X, connectivity=None, distance_thresholds=None):
    """
    Computes labels from ward hierarchical clustering at multiple distance thresholds,
    without recomputing the tree at each threshold.
    """
    children, _, n_leaves, _, distances = ward_tree(
        X, connectivity=connectivity, return_distance=True
    )

    labels_by_distance = []
    for distance_threshold in distance_thresholds:
        n_clusters_ = np.count_nonzero(distances >= distance_threshold) + 1
        labels_at_d = _hc_cut(n_clusters_, children, n_leaves)

        labels_by_distance.append(labels_at_d)

    labels_by_distance = np.stack(labels_by_distance, axis=1)
    return labels_by_distance


def agglomerate_mesh(mesh, features, distance_thresholds=None) -> np.ndarray:
    if not (np.isfinite(features)).all():
        return None
    else:
        submesh_adj = mesh_to_adjacency(mesh)
        submesh_adj = submesh_adj + submesh_adj.T
        submesh_connectivity = submesh_adj > 0
        labels_by_distance = multicut_ward(
            features,
            connectivity=submesh_connectivity,
            distance_thresholds=distance_thresholds,
        )
        return labels_by_distance


def fix_split_labels(agg_labels, distance_thresholds):
    cols = [f"distance_{d}" for d in distance_thresholds] + ["submesh"]
    agg_labels_df = pd.DataFrame(agg_labels, columns=cols).astype(int)

    good_agg_labels_df = agg_labels_df[
        agg_labels_df[f"distance_{distance_thresholds[-1]}"] != -1
    ].copy()

    shifts = good_agg_labels_df.groupby("submesh", sort=True).max().cumsum()

    for col in good_agg_labels_df.columns:
        if col == "submesh":
            continue
        good_agg_labels_df[col] += good_agg_labels_df["submesh"].map(shifts[col])

    agg_labels_df = good_agg_labels_df.reindex(agg_labels_df.index, fill_value=-1)

    return agg_labels_df.values


def agglomerate_split_mesh(
    splitter: MeshStitcher, features: np.ndarray, distance_thresholds: list
):
    agg_labels = splitter.apply_on_features(
        agglomerate_mesh,
        features,
        distance_thresholds=distance_thresholds,
        add_label_column=True,
        fill_value=-1,
    )
    agg_labels = fix_split_labels(agg_labels, distance_thresholds)

    return agg_labels


def aggregate_features(features, labels, func="mean"):
    feature_df = pd.DataFrame(features)
    cols = feature_df.columns
    feature_df["label"] = labels
    agg_feature_df = feature_df.groupby("label").agg(func=func)
    return agg_feature_df[cols].values
