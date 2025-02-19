import warnings
from typing import Union

import numpy as np
import pandas as pd
from sklearn.cluster import ward_tree

# TODO dangerous to import private function here
from sklearn.cluster._agglomerative import _hc_cut

from .split import MeshStitcher
from .utils import mesh_to_adjacency, subset_mesh_by_indices


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
        if np.any(labels_at_d == -1):
            raise ValueError("Some labels are -1")
        labels_by_distance.append(labels_at_d)

    labels_by_distance = np.stack(labels_by_distance, axis=1)
    return labels_by_distance


def agglomerate_mesh(mesh, features, distance_thresholds=None) -> np.ndarray:
    """this is the version that fixes issues where only some parts of a mesh are nan"""
    if not (np.isfinite(features)).any():
        return None
    else:
        features = features.copy()
        # features[mask] = 1000000
        # TODO masking out nans here is definitely a hack
        # alternatively could agglomerate on the underlying mesh with features in one
        # go, this shouldn't even be a problem then.
        mask = np.isfinite(features).all(axis=1)
        indices = np.arange(len(features))[mask]

        submesh = subset_mesh_by_indices(mesh, indices)
        subfeatures = features[mask]

        submesh_adj = mesh_to_adjacency(submesh)
        submesh_adj = submesh_adj + submesh_adj.T
        submesh_connectivity = submesh_adj > 0
        # there's an issue here with infilling the connectivity
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels_by_distance = multicut_ward(
                subfeatures,
                connectivity=submesh_connectivity,
                distance_thresholds=distance_thresholds,
            )
        labels_by_distance_full = np.full(
            (len(features), len(distance_thresholds)), -1, dtype=int
        )
        labels_by_distance_full[mask] = labels_by_distance
        return labels_by_distance_full


# def agglomerate_mesh(mesh, features, distance_thresholds=None) -> np.ndarray:
#     if not (np.isfinite(features)).all():
#         return None
#     else:
#         features = features.copy()

#         submesh_adj = mesh_to_adjacency(mesh)
#         submesh_adj = submesh_adj + submesh_adj.T
#         submesh_connectivity = submesh_adj > 0

#         labels_by_distance = multicut_ward(
#             features,
#             connectivity=submesh_connectivity,
#             distance_thresholds=distance_thresholds,
#         )

#         return labels_by_distance


def fix_split_labels(agg_labels, submesh_mapping):
    for label_column in range(agg_labels.shape[1]):
        valid_mask = agg_labels[:, label_column] != -1

        valid_labels = agg_labels[valid_mask, label_column]
        submesh_labels = submesh_mapping[valid_mask]
        
        tuple_labels = pd.Index(list(zip(valid_labels, submesh_labels)))
        unique_labels = np.unique(tuple_labels)

        new_unique_labels = np.arange(len(unique_labels))
        label_mapping = dict(zip(unique_labels, new_unique_labels))
        new_labels = np.array([label_mapping[label] for label in tuple_labels])

        agg_labels[valid_mask, label_column] = new_labels

    return agg_labels


def agglomerate_split_mesh(
    splitter: MeshStitcher,
    features: np.ndarray,
    distance_thresholds: Union[list, int, float],
):
    if isinstance(distance_thresholds, (int, float)):
        distance_thresholds = [distance_thresholds]
        was_single = True
    else:
        was_single = False
    agg_labels = splitter.apply_on_features(
        agglomerate_mesh,
        features,
        distance_thresholds=distance_thresholds,
        # add_label_column=True,
        fill_value=-1,
    )
    agg_labels = agg_labels.astype(int)
    agg_labels = fix_split_labels(agg_labels, splitter.submesh_mapping)

    if was_single:
        return agg_labels[:, 0]

    return agg_labels


def aggregate_features(features, labels, weights=None, func="mean") -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        feature_df = pd.DataFrame(features)
    else:
        feature_df = features
    cols = feature_df.columns
    if labels is None:
        return feature_df
    feature_df["label"] = labels
    if func == "mean" and weights is not None:
        feature_df["weight"] = weights
        agg_feature_df = (
            feature_df.groupby("label")
            .apply(
                lambda x: pd.Series(
                    np.average(x, weights=feature_df.loc[x.index, "weight"], axis=0),
                    index=x.columns,
                ),
                include_groups=False,
            )
            .drop(columns="weight")
        )
    else:
        agg_feature_df = feature_df.groupby("label").agg(func=func)
    out = agg_feature_df[cols]
    return out


def blow_up_features(agg_features_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    out = agg_features_df.loc[labels]
    out.reset_index(drop=True, inplace=True)
    return out
