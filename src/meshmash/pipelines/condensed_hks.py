import time
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from ..agglomerate import (
    condense_features,
    fix_split_labels_and_features,
)
from ..decompose import compute_hks
from ..graph import condense_mesh_to_graph
from ..simplify import simplify_mesh, simplify_to_density
from ..split import MeshStitcher
from ..types import interpret_mesh
from ..utils import (
    compute_distances_to_point,
    expand_labels,
    threshold_mesh_by_component_size,
)


class CondensedHKSResult(NamedTuple):
    simple_mesh: tuple
    mapping: np.ndarray
    stitcher: MeshStitcher
    simple_labels: np.ndarray
    labels: np.ndarray
    condensed_features: pd.DataFrame
    condensed_nodes: pd.DataFrame
    condensed_edges: pd.DataFrame
    timing_info: dict


def compute_condensed_hks(
    mesh,
    n_components=32,
    t_min=5e4,
    t_max=2e7,
    max_eigenvalue=1e-5,
    robust=True,
    mollify_factor=1e-5,
    truncate_extra=True,
    drop_first=True,
    decomposition_dtype=np.float32,
    compute_hks_kwargs: dict = {},
    distance_threshold=3.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute HKS features and aggregate them on a single (unsplit) mesh.

    This is a lightweight helper used internally by
    [condensed_hks_pipeline][meshmash.pipelines.condensed_hks.condensed_hks_pipeline] to process individual submeshes.  For
    large meshes, use the pipeline functions instead.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    n_components :
        Number of HKS timescales.
    t_min :
        Minimum diffusion timescale.
    t_max :
        Maximum diffusion timescale.
    max_eigenvalue :
        Maximum Laplacian eigenvalue for the HKS computation.
    robust :
        If ``True``, use the robust Laplacian.
    mollify_factor :
        Mollification factor for the robust Laplacian.
    truncate_extra :
        If ``True``, discard eigenpairs past ``max_eigenvalue``.
    drop_first :
        If ``True``, drop the first (area-proportional) eigenpair.
    decomposition_dtype :
        Floating-point dtype for the eigendecomposition.
    compute_hks_kwargs :
        Extra keyword arguments forwarded to
        [compute_hks][meshmash.decompose.compute_hks].
    distance_threshold :
        Ward linkage-distance threshold for agglomeration.

    Returns
    -------
    condensed_features :
        Per-domain aggregated HKS feature DataFrame indexed by domain
        label (including ``-1`` for unassigned vertices).
    labels :
        Per-vertex domain label array of length ``V``.
    """
    X_hks = compute_hks(
        mesh,
        n_components=n_components,
        t_min=t_min,
        t_max=t_max,
        max_eigenvalue=max_eigenvalue,
        robust=robust,
        mollify_factor=mollify_factor,
        truncate_extra=truncate_extra,
        drop_first=drop_first,
        decomposition_dtype=decomposition_dtype,
        **compute_hks_kwargs,
    )

    return condense_features(
        mesh,
        pd.DataFrame(X_hks, columns=[f"hks_{i}" for i in range(X_hks.shape[1])]),
        distance_threshold=distance_threshold,
    )


def compute_split_condensed_hks(
    mesh,
    overlap_distance=20_000,
    max_vertex_threshold=20_000,
    min_vertex_threshold=200,
    max_overlap_neighbors=60_000,
    n_components=32,
    t_min=5e4,
    t_max=2e7,
    max_eigenvalue=1e-5,
    robust=True,
    mollify_factor=1e-5,
    truncate_extra=True,
    drop_first=True,
    decomposition_dtype="float32",
    compute_hks_kwargs: dict = {},
    distance_threshold=3.0,
    n_jobs: Optional[int] = -1,
    verbose=False,
) -> tuple[pd.DataFrame, np.ndarray, MeshStitcher]:
    """Split a mesh into chunks, and condense each chunk's HKS onto local domains.

    The chunked middle of [condensed_hks_pipeline][meshmash.pipelines.condensed_hks.condensed_hks_pipeline],
    on its own: spectral bisection into overlapping chunks, per-chunk
    [compute_condensed_hks][meshmash.pipelines.condensed_hks.compute_condensed_hks],
    and reconciliation of the per-chunk domain labels into one global
    numbering.  Aggregating *within* each chunk before stitching is what keeps
    memory proportional to a chunk rather than to the whole mesh.

    It takes the mesh as given.  Neither component thresholding nor
    simplification happens here, so a caller that has already conditioned its
    mesh gets exactly this and nothing more — which is what the caller cannot
    get by passing ``simplify_target_reduction=None`` to the pipeline, since
    that switches off the simplification but not the component threshold.

    It also stops at the aggregated features: no condensed node or edge table.
    Those are computed on the pre-threshold mesh, which this function is not
    handed.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
        Conditioned as the caller wants it; nothing here removes vertices.
    overlap_distance :
        Geodesic radius used to grow each chunk into its overlap region.
    max_vertex_threshold :
        Maximum vertices per core chunk before overlapping.
    min_vertex_threshold :
        Minimum connected-component size within
        [split_mesh][meshmash.split.MeshStitcher.split_mesh].
    max_overlap_neighbors :
        Cap on overlap region size (number of nearest neighbours);
        overrides ``overlap_distance`` when set.
    n_components :
        Number of HKS timescales.
    t_min :
        Minimum diffusion timescale.
    t_max :
        Maximum diffusion timescale.
    max_eigenvalue :
        Maximum Laplacian eigenvalue used in the HKS computation.
    robust :
        If ``True``, use the robust Laplacian for HKS (recommended).
    mollify_factor :
        Mollification factor for the robust Laplacian.
    truncate_extra :
        If ``True``, discard eigenpairs that overshoot ``max_eigenvalue``.
    drop_first :
        If ``True``, drop the first (area-proportional) eigenpair.
    decomposition_dtype :
        Floating-point dtype for the eigendecomposition.
    compute_hks_kwargs :
        Extra keyword arguments forwarded to
        [compute_hks][meshmash.decompose.compute_hks].
    distance_threshold :
        Ward linkage-distance threshold used to cut the agglomeration tree
        into local domains.
    n_jobs :
        Number of parallel workers for [Parallel][joblib.Parallel].
    verbose :
        Verbosity level.

    Returns
    -------
    condensed_features :
        Log of the area-weighted mean HKS per domain, indexed by global domain
        label.  Includes a row for the null label ``-1``, whose values are all
        NaN.
    labels :
        Per-vertex domain label array of length ``V``.  ``-1`` for a vertex in
        no domain.
    stitcher :
        The fitted [MeshStitcher][meshmash.split.MeshStitcher].
    """
    stitcher = MeshStitcher(mesh, n_jobs=n_jobs, verbose=verbose)
    stitcher.split_mesh(
        overlap_distance=overlap_distance,
        max_vertex_threshold=max_vertex_threshold,
        min_vertex_threshold=min_vertex_threshold,
        max_overlap_neighbors=max_overlap_neighbors,
        verify_connected=False,
    )

    if verbose:
        print("Computing HKS across submeshes...")

    results_by_submesh = stitcher.apply(
        compute_condensed_hks,
        n_components=n_components,
        t_min=t_min,
        t_max=t_max,
        max_eigenvalue=max_eigenvalue,
        robust=robust,
        mollify_factor=mollify_factor,
        truncate_extra=truncate_extra,
        drop_first=drop_first,
        decomposition_dtype=decomposition_dtype,
        compute_hks_kwargs=compute_hks_kwargs,
        distance_threshold=distance_threshold,
        stitch=False,
    )
    sub_agg_labels = stitcher.stitch_features(
        [result[1] for result in results_by_submesh],
        fill_value=-1,
    ).reshape(-1)
    data_by_submesh = [res[0] for res in results_by_submesh]

    agg_labels, condensed_hks_df = fix_split_labels_and_features(
        sub_agg_labels,
        stitcher.submesh_mapping,
        data_by_submesh,
    )
    condensed_hks_df = np.log(condensed_hks_df)

    return condensed_hks_df, agg_labels, stitcher


def condensed_hks_pipeline(
    mesh,
    simplify_agg=7,
    simplify_target_reduction=0.7,
    simplify_target_density=None,
    overlap_distance=20_000,
    max_vertex_threshold=20_000,
    min_vertex_threshold=200,
    max_overlap_neighbors=60_000,
    n_components=32,
    t_min=5e4,
    t_max=2e7,
    max_eigenvalue=1e-5,
    robust=True,
    mollify_factor=1e-5,
    truncate_extra=True,
    drop_first=True,
    decomposition_dtype="float32",
    compute_hks_kwargs: dict = {},
    nuc_point=None,
    distance_threshold=3.0,
    auxiliary_features=True,
    n_jobs: Optional[int] = -1,
    verbose=False,
) -> CondensedHKSResult:
    """Compute HKS features and produce a condensed node-edge graph of a mesh.

    This is the primary entry point for the HKS pipeline.  It is more
    memory-efficient than [chunked_hks_pipeline][meshmash.pipelines.chunked_hks.chunked_hks_pipeline] because features are
    aggregated *within* each submesh chunk before the results are combined,
    rather than stitching the full per-vertex feature matrix first.

    Parameters
    ----------
    mesh :
        Input mesh.  Either a ``(vertices, faces)`` tuple or an object
        with ``vertices`` and ``faces`` attributes.
    simplify_agg :
        Decimation aggressiveness (0–10).  Higher values are faster but
        reduce mesh quality.  Low values may prevent reaching
        ``simplify_target_reduction``.
    simplify_target_reduction :
        Fraction of triangles to remove during simplification.  ``None``
        skips simplification.  Mutually exclusive with
        ``simplify_target_density``: to use density-targeted simplification,
        set this to ``None``.
    simplify_target_density :
        Target vertex density (vertices per unit surface area) for
        [simplify_to_density][meshmash.simplify.simplify_to_density].  Unlike
        the relative ``simplify_target_reduction``, this yields the same
        physical mesh resolution regardless of dataset units or size.  Only
        one of ``simplify_target_reduction`` / ``simplify_target_density``
        may be non-``None``; providing both raises ``ValueError``.
    overlap_distance :
        Geodesic radius used to grow each chunk into its overlap region.
    max_vertex_threshold :
        Maximum vertices per core chunk before overlapping.
    min_vertex_threshold :
        Minimum component size; smaller components are discarded.
    max_overlap_neighbors :
        Cap on overlap region size (number of nearest neighbours);
        overrides ``overlap_distance`` when set.
    n_components :
        Number of HKS timescales.  Timescales are log-spaced between
        ``t_min`` and ``t_max`` and determine the number of HKS features
        per vertex.
    t_min :
        Minimum diffusion timescale.
    t_max :
        Maximum diffusion timescale.
    max_eigenvalue :
        Maximum Laplacian eigenvalue used in the HKS computation.
    robust :
        If ``True``, use the robust Laplacian for HKS (recommended).
    mollify_factor :
        Mollification factor for the robust Laplacian.
    truncate_extra :
        If ``True``, discard eigenpairs that overshoot ``max_eigenvalue``.
    drop_first :
        If ``True``, drop the first (area-proportional) eigenpair.
    decomposition_dtype :
        Floating-point dtype for the eigendecomposition.
    compute_hks_kwargs :
        Extra keyword arguments forwarded to
        [compute_hks][meshmash.decompose.compute_hks].
    nuc_point :
        Coordinates of the nucleus/reference point.  If provided, a
        ``distance_to_nucleus`` column is added to the condensed node table.
    distance_threshold :
        Ward linkage-distance threshold used to cut the agglomeration tree
        into local domains.
    auxiliary_features :
        If ``True``, append condensed node-table columns (centroid, area,
        etc.) to the aggregated feature DataFrame.
    n_jobs :
        Number of parallel workers for [Parallel][joblib.Parallel].  ``-1``
        uses all available cores.
    verbose :
        Verbosity level.

    Returns
    -------
    simple_mesh :
        The simplified mesh as a ``(vertices, faces)`` tuple.
    mapping :
        Array of length ``V_original`` mapping each original vertex to its
        index in the simplified mesh.  ``-1`` for discarded vertices.
    stitcher :
        Fitted [MeshStitcher][meshmash.split.MeshStitcher] for the simplified
        mesh.
    simple_labels :
        Per-vertex domain label array for the simplified mesh.
    labels :
        Per-vertex domain label array for the *original* mesh.
    condensed_features :
        Per-domain aggregated feature DataFrame (log-HKS + optional
        auxiliary features), indexed by domain label.
    condensed_nodes :
        Node table of the condensed mesh graph; see
        [condense_mesh_to_graph][meshmash.graph.condense_mesh_to_graph].
    condensed_edges :
        Edge table of the condensed mesh graph.
    timing_info :
        Dictionary with wall-clock times (seconds) for each pipeline step.

    Notes
    -----
    This pipeline consists of the following steps:

    1. Mesh simplification via ``fast-simplification``.
    2. Spectral bisection into overlapping chunks
       ([MeshStitcher][meshmash.split.MeshStitcher]).
    3. Per-chunk: [compute_hks][meshmash.decompose.compute_hks], Ward
       agglomeration, and area-weighted aggregation
       ([compute_condensed_hks][meshmash.pipelines.condensed_hks.compute_condensed_hks]).  Aggregating *before* stitching
       keeps memory use proportional to the chunk size rather than the
       full mesh.
    4. Global label reconciliation across chunks
       ([fix_split_labels][meshmash.agglomerate.fix_split_labels]).
    5. Assembly of the condensed node-edge graph
       ([condense_mesh_to_graph][meshmash.graph.condense_mesh_to_graph]).
    """
    timing_info = {}
    starttime = time.time()

    # input mesh
    original_mesh = interpret_mesh(mesh)

    mesh, indices_from_original = threshold_mesh_by_component_size(
        original_mesh, size_threshold=min_vertex_threshold
    )

    # mesh simplification
    if simplify_target_reduction is not None and simplify_target_density is not None:
        raise ValueError(
            "Provide only one of `simplify_target_reduction` or "
            "`simplify_target_density`, not both. To use density-targeted "
            "simplification, set `simplify_target_reduction=None`."
        )
    if simplify_target_density is not None:
        vertices, faces, thresh_to_simple_mapping = simplify_to_density(
            mesh,
            target_density=simplify_target_density,
            simplify_agg=simplify_agg,
            verbose=verbose,
        )
        mesh = (vertices, faces)
    else:
        # `simplify_mesh` also covers `target_reduction=None`, which returns
        # the mesh untouched with an identity mapping.
        mesh, thresh_to_simple_mapping = simplify_mesh(
            mesh, agg=simplify_agg, target_reduction=simplify_target_reduction
        )

    # mesh splitting, HKS, agglomeration, and aggregation
    currtime = time.time()
    condensed_hks_df, simple_agg_labels, stitcher = compute_split_condensed_hks(
        mesh,
        overlap_distance=overlap_distance,
        max_vertex_threshold=max_vertex_threshold,
        min_vertex_threshold=min_vertex_threshold,
        max_overlap_neighbors=max_overlap_neighbors,
        n_components=n_components,
        t_min=t_min,
        t_max=t_max,
        max_eigenvalue=max_eigenvalue,
        robust=robust,
        mollify_factor=mollify_factor,
        truncate_extra=truncate_extra,
        drop_first=drop_first,
        decomposition_dtype=decomposition_dtype,
        compute_hks_kwargs=compute_hks_kwargs,
        distance_threshold=distance_threshold,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    timing_info["hks_time"] = time.time() - currtime

    # reconstruct mapping to original mesh
    mapping = np.full(len(original_mesh[0]), -1, dtype=np.int32)
    mapping[indices_from_original] = thresh_to_simple_mapping

    agg_labels = expand_labels(simple_agg_labels, mapping)

    # condense mesh to graph, compute some additional auxiliary features
    condensed_node_table, condensed_edge_table = condense_mesh_to_graph(
        original_mesh, agg_labels, add_component_features=True
    )

    if auxiliary_features:
        if nuc_point is not None:
            condensed_node_table["distance_to_nucleus"] = compute_distances_to_point(
                condensed_node_table[["x", "y", "z"]].values, nuc_point
            ).astype(np.float32)
        else:
            condensed_node_table["distance_to_nucleus"] = np.full(
                condensed_node_table.shape[0], np.nan, dtype=np.float32
            )

    # for consistency with the rest of the pipeline, make sure null label is present
    assert -1 in condensed_hks_df.index

    timing_info["pipeline_time"] = time.time() - starttime
    out = CondensedHKSResult(
        mesh,
        mapping,
        stitcher,
        simple_agg_labels,
        agg_labels,
        condensed_hks_df,
        condensed_node_table,
        condensed_edge_table,
        timing_info,
    )

    return out
