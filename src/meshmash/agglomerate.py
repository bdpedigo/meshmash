import logging
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.cluster import ward_tree

# TODO dangerous to import private function here
from sklearn.cluster._agglomerative import _hc_cut

from .split import MeshStitcher
from .types import ArrayLike
from .utils import mesh_to_adjacency, subset_mesh_by_indices


def multicut_ward(
    X: ArrayLike,
    connectivity: Optional[sparse.sparray] = None,
    distance_thresholds: Optional[list[float]] = None,
) -> np.ndarray:
    """Compute Ward cluster labels at multiple distance thresholds.

    Builds the Ward linkage tree once with
    :func:`sklearn.cluster.ward_tree`, then cuts it at each threshold in
    ``distance_thresholds`` without rebuilding.  This is more efficient
    than fitting a separate :class:`sklearn.cluster.AgglomerativeClustering`
    for each threshold.

    Parameters
    ----------
    X :
        Feature matrix of shape ``(N, F)``.
    connectivity :
        Optional sparse connectivity matrix of shape ``(N, N)``
        constraining which samples may be merged (passed to
        :func:`sklearn.cluster.ward_tree`).
    distance_thresholds :
        List of linkage-distance thresholds at which to cut the tree.
        Each threshold yields one column in the output.

    Returns
    -------
    :
        Integer label array of shape ``(N, len(distance_thresholds))``.
        Column ``i`` contains cluster labels for threshold ``i``.
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
    """Apply connectivity-constrained Ward clustering to vertex features on a mesh.

    Handles vertices with non-finite feature values by masking them out and
    running clustering only on the valid sub-mesh, then filling ``-1`` back
    into the invalid positions.

    Parameters
    ----------
    mesh :
        Input mesh accepted by :func:`~meshmash.types.interpret_mesh`.
    features :
        Per-vertex feature matrix of shape ``(V, F)``.
    distance_thresholds :
        Single threshold or list of thresholds passed to
        :func:`multicut_ward`.  ``None`` assigns each vertex a unique
        label (i.e. no clustering).

    Returns
    -------
    :
        Integer label array of shape ``(V, T)`` where ``T`` is the
        number of thresholds.  Vertices with non-finite features receive
        label ``-1``.  Returns ``None`` if no vertices have finite
        features or the sub-mesh has no faces.
    """
    if isinstance(distance_thresholds, (int, float)) or distance_thresholds is None:
        distance_thresholds = [distance_thresholds]
    if not (np.isfinite(features).all(axis=1)).any():
        return None
    elif len(distance_thresholds) == 1 and distance_thresholds[0] is None:
        return np.arange(len(features)).reshape(-1, 1)
    else:
        features = features.copy()
        # features[mask] = 1000000
        # TODO masking out nans here is definitely a hack
        # alternatively could agglomerate on the underlying mesh with features in one
        # go, this shouldn't even be a problem then.
        mask = np.isfinite(features).all(axis=1)
        indices = np.arange(len(features))[mask]

        submesh = subset_mesh_by_indices(mesh, indices)
        if len(submesh[1]) == 0:
            return None

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


def fix_split_labels(agg_labels: np.ndarray, submesh_mapping: np.ndarray) -> np.ndarray:
    """Remap per-submesh local labels to globally unique integers.

    When each submesh independently assigns cluster labels ``0, 1, 2, …``,
    the same integer can refer to different clusters in different submeshes.
    This function treats each ``(submesh_id, local_label)`` pair as a
    unique cluster and remaps them to a single contiguous range
    ``0, 1, 2, …``.

    Parameters
    ----------
    agg_labels :
        Per-vertex label array of shape ``(V, T)`` where ``T`` is the
        number of distance thresholds.  Vertices not belonging to any
        submesh have label ``-1`` and are left unchanged.
    submesh_mapping :
        Per-vertex integer array of length ``V`` indicating which submesh
        each vertex belongs to (``-1`` for unassigned vertices).

    Returns
    -------
    :
        Modified ``agg_labels`` with globally unique integers, same shape
        as the input.
    """
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


def fix_split_labels_and_features(
    agg_labels: np.ndarray,
    submesh_mapping: np.ndarray,
    features_by_submesh: list[pd.DataFrame],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Remap per-submesh labels to globally unique integers and align feature DataFrames.

    Like :func:`fix_split_labels`, but also re-indexes the per-submesh
    feature DataFrames so that their row indices match the new global label
    integers and concatenates them into a single DataFrame.

    Parameters
    ----------
    agg_labels :
        Per-vertex label array of length ``V``.  Vertices not belonging to
        any submesh have label ``-1`` and are left unchanged.
    submesh_mapping :
        Per-vertex integer array of length ``V`` indicating which submesh
        each vertex belongs to (``-1`` for unassigned vertices).
    features_by_submesh :
        List of per-submesh feature DataFrames, one per submesh, indexed
        by local cluster label.

    Returns
    -------
    agg_labels :
        Modified ``agg_labels`` with globally unique integers, same length
        as the input.
    condensed_features :
        Concatenated feature DataFrame indexed by global label, including
        a row for the null label ``-1``.
    """
    valid_mask = agg_labels != -1

    valid_labels = agg_labels[valid_mask]
    submesh_labels = submesh_mapping[valid_mask]

    tuple_labels = pd.Index(list(zip(submesh_labels, valid_labels)))
    unique_labels = np.unique(tuple_labels)

    new_unique_labels = np.arange(len(unique_labels))
    label_mapping = dict(zip(unique_labels, new_unique_labels))
    label_mapping_series = pd.Series(label_mapping)
    new_labels = np.array([label_mapping[label] for label in tuple_labels])

    agg_labels[valid_mask] = new_labels

    new_data = []
    for submesh_index, data in enumerate(features_by_submesh):
        data: pd.DataFrame
        data.drop(-1, inplace=True, errors="ignore")
        if submesh_index in label_mapping_series.index.get_level_values(0):
            sub_label_mapping = label_mapping_series.loc[submesh_index]
            data.index = data.index.map(sub_label_mapping)
            data.drop(data.index[data.index.isna()], inplace=True)
            data.index = data.index.astype(int)
            new_data.append(data)
        else:
            continue

    empty_data = pd.DataFrame(columns=data.columns, index=[-1])
    new_data = pd.concat([empty_data] + new_data)

    assert np.isin(np.unique(agg_labels), new_data.index).all()

    return agg_labels, new_data


def agglomerate_split_mesh(
    splitter: MeshStitcher,
    features: np.ndarray,
    distance_thresholds: Union[list, int, float],
) -> np.ndarray:
    """Apply Ward clustering across submeshes and return globally unique labels.

    Applies :func:`agglomerate_mesh` to each submesh via
    :meth:`~meshmash.split.MeshStitcher.apply_on_features`, then calls
    :func:`fix_split_labels` to make the cluster labels globally unique
    across all submeshes.

    Parameters
    ----------
    splitter :
        A fitted :class:`~meshmash.split.MeshStitcher`.
    features :
        Per-vertex feature matrix of shape ``(V, F)`` for the full mesh.
    distance_thresholds :
        Single threshold or list of thresholds passed to
        :func:`multicut_ward`.

    Returns
    -------
    :
        Integer label array of shape ``(V,)`` if a single threshold was
        given, or ``(V, T)`` for a list of ``T`` thresholds.  Unassigned
        vertices have label ``-1``.
    """
    if isinstance(distance_thresholds, (int, float)) or distance_thresholds is None:
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


def aggregate_features(
    features: Union[np.ndarray, pd.DataFrame],
    labels: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    func: str = "mean",
) -> pd.DataFrame:
    """Aggregate per-vertex features to per-label summaries.

    Groups vertices by ``labels`` and applies ``func`` (or an
    area-weighted mean when ``weights`` is provided) to produce one row
    per unique label.  The result is reindexed to include every integer
    from ``-1`` to ``labels.max()``, inserting ``NaN`` for any missing
    labels.

    Parameters
    ----------
    features :
        Per-vertex feature matrix of shape ``(V, F)``, as an array or
        DataFrame.
    labels :
        Integer label array of length ``V``.  ``-1`` is treated as the
        null label.  If ``None``, the features are returned unchanged.
    weights :
        Per-vertex weight array of length ``V`` used for area-weighted
        aggregation when ``func="mean"``.  ``None`` falls back to an
        unweighted mean.
    func :
        Aggregation function name recognised by
        :meth:`pandas.DataFrameGroupBy.agg` (e.g. ``"mean"``,
        ``"median"``).  Ignored when ``weights`` is provided.

    Returns
    -------
    :
        DataFrame of shape ``(labels.max() + 2, F)`` indexed from
        ``-1`` to ``labels.max()``, where each row contains the
        aggregated features for that label.
    """
    if not isinstance(features, pd.DataFrame):
        feature_df = pd.DataFrame(features)
    else:
        feature_df = features.copy()
    cols = feature_df.columns
    if labels is None:
        return feature_df
    feature_df["label"] = labels
    if func == "mean" and weights is not None:
        feature_df["weight"] = weights

        def _weighted_average(x):
            weights = feature_df.loc[x.index, "weight"]
            if weights.sum() == 0:
                weights = None
                logging.warning(
                    "Weights sum to zero for a group, using unweighted average in aggregation."
                )
            out = pd.Series(
                np.average(x, weights=weights, axis=0),
                index=x.columns,
            )
            return out

        agg_feature_df = (
            feature_df.groupby("label")
            .apply(
                _weighted_average,
                include_groups=False,
            )
            .drop(columns="weight")
        )
    else:
        agg_feature_df = feature_df.groupby("label").agg(func=func)

    expected_indices = np.arange(-1, labels.max() + 1)
    agg_feature_df = agg_feature_df.reindex(expected_indices, copy=False)

    out = agg_feature_df[cols]
    return out


def blow_up_features(agg_features_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Expand per-label aggregated features back to per-vertex features.

    Inverse of :func:`aggregate_features`.  Looks up each vertex's label
    in ``agg_features_df`` to produce a per-vertex DataFrame.

    Parameters
    ----------
    agg_features_df :
        Per-label feature DataFrame indexed by label integer, as returned
        by :func:`aggregate_features`.
    labels :
        Per-vertex label array of length ``V``.

    Returns
    -------
    :
        Per-vertex feature DataFrame of shape ``(V, F)`` with a reset
        integer index.  Vertices with label ``-1`` receive ``NaN`` values.
    """
    agg_features_df = agg_features_df.copy()
    if -1 not in agg_features_df.index:
        agg_features_df.loc[-1] = np.nan
    out = agg_features_df.loc[labels]
    out.reset_index(drop=True, inplace=True)
    return out
