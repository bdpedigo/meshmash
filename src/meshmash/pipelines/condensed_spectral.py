"""Three descriptor families off one eigendecomposition per chunk, condensed.

[condensed_hks][meshmash.pipelines.condensed_hks] computes one family, the heat
kernel signature.  This computes three — the HKS, the diffused curvature
invariants, and the diffused normal-tensor invariants — from the *same*
eigendecomposition, so the second and third families cost another matrix
product per band rather than another solve.  Everything else is the HKS
pipeline's shape: spectral bisection into overlapping chunks, per-chunk
featurizing and agglomeration, then reconciliation of the per-chunk domain
numbering into one global one.

**Two timescale grids, and that is the point of the fusing.**  The kernel
diagonal is read at the ``n_components`` HKS timescales, while the curvature
and tensor channels are diffused at ``n_scales`` timescales over the same
``[t_min, t_max]`` range.  The HKS wants a fine grid because its columns are
cheap; the signal channels carry ten columns each, so they want a coarse one.
[compute_diffused_curvature][meshmash.curvature.compute_diffused_curvature]
takes both because a single decomposition serves them both.

**Only the HKS decides the domains.**  The agglomeration cuts on the ``hks_``
columns alone, and every column is aggregated onto the domains that cut
produces.  So adding a family changes what each domain *says* and not where
the domains are, which is what makes a run with the extra families comparable
to a run without them.  The other two families could not cut anyway: Ward runs
on the log, and ``normal_`` fractions reaching zero and the signed
``curvature_mean_`` and ``curvature_k`` columns have no log.
"""

import time
from typing import Optional

import numpy as np
import pandas as pd

from ..agglomerate import condense_features, fix_split_labels_and_features
from ..curvature import compute_diffused_curvature, diffused_curvature_feature_names
from ..decompose import get_hks_filter
from ..split import MeshStitcher

#: How many timescales the curvature and tensor channels are diffused at, when
#: the caller does not say.  Eight against the HKS's thirty-two: each of these
#: scales carries ten columns where an HKS scale carries one.
DEFAULT_N_SCALES = 8


def hks_column_names(n_components: int) -> list[str]:
    """The ``hks_`` columns of a condensed spectral table, in order.

    The block the agglomeration cuts on and the only block that is logged.
    Derived from
    [diffused_curvature_feature_names][meshmash.curvature.diffused_curvature_feature_names]
    rather than rebuilt, so the two cannot drift apart.

    Parameters
    ----------
    n_components :
        Number of HKS timescales.

    Returns
    -------
    :
        ``["hks_0", ..., f"hks_{n_components - 1}"]``.
    """
    return diffused_curvature_feature_names(0, n_diagonal=n_components)[:n_components]


def compute_condensed_spectral(
    mesh,
    n_components: int = 32,
    n_scales: int = DEFAULT_N_SCALES,
    t_min: float = 5e4,
    t_max: float = 2e7,
    max_eigenvalue: float = 1e-5,
    robust: bool = True,
    mollify_factor: float = 1e-5,
    truncate_extra: bool = True,
    drop_first: bool = True,
    decomposition_dtype=np.float32,
    compute_diffused_curvature_kwargs: dict = {},
    distance_threshold: float = 3.0,
    seed: Optional[int] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Featurize and condense a single (unsplit) mesh, three families at once.

    The counterpart of
    [compute_condensed_hks][meshmash.pipelines.condensed_hks.compute_condensed_hks]:
    a lightweight helper that
    [compute_split_condensed_spectral][meshmash.pipelines.condensed_spectral.compute_split_condensed_spectral]
    runs on each chunk.  For a large mesh, use that instead.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    n_components :
        Number of HKS timescales, for the kernel diagonal.
    n_scales :
        Number of diffusion timescales, for the curvature and tensor channels.
        Spaced logarithmically over the same ``[t_min, t_max]`` range as the
        HKS grid, so a scale index means one physical scale throughout.
    t_min :
        Smallest diffusion timescale, for both grids.
    t_max :
        Largest diffusion timescale, for both grids.
    max_eigenvalue :
        Maximum Laplacian eigenvalue to decompose up to.
    robust :
        If ``True``, use the robust Laplacian.
    mollify_factor :
        Mollification factor for the robust Laplacian.
    truncate_extra :
        If ``True``, discard eigenpairs past ``max_eigenvalue``.
    drop_first :
        If ``True``, drop the constant eigenpair from the kernel diagonal.
        The signal channels are compensated for it either way.
    decomposition_dtype :
        Floating-point dtype for the eigendecomposition.
    compute_diffused_curvature_kwargs :
        Extra keyword arguments forwarded to
        [compute_diffused_curvature][meshmash.curvature.compute_diffused_curvature].
    distance_threshold :
        Ward linkage-distance threshold for the agglomeration.
    seed :
        Seed for the ARPACK starting vector.  ``None`` lets ARPACK draw its
        own, so two runs of this pipeline on the same mesh differ at
        ``decomposition_dtype`` — and Ward flips merges on those ties, so the
        domain count moves too.  An integer makes the whole pipeline
        reproducible.  Every chunk is seeded alike, which is harmless: the
        starting vector only has to overlap the wanted subspace.

    Returns
    -------
    condensed_features :
        Per-domain aggregated features indexed by domain label, including the
        null label ``-1``.  Columns as
        [diffused_curvature_feature_names][meshmash.curvature.diffused_curvature_feature_names]
        gives them.  Physical values, not logged — see
        [compute_split_condensed_spectral][meshmash.pipelines.condensed_spectral.compute_split_condensed_spectral],
        which logs the ``hks_`` block once every chunk is stitched.
    labels :
        Per-vertex domain label array of length ``V``.
    """
    features = compute_diffused_curvature(
        mesh,
        np.geomspace(t_min, t_max, n_scales),
        diagonal_filter=get_hks_filter(
            t_max, t_min, n_components, dtype=decomposition_dtype
        ),
        max_eigenvalue=max_eigenvalue,
        truncate_extra=truncate_extra,
        drop_first=drop_first,
        robust=robust,
        mollify_factor=mollify_factor,
        decomposition_dtype=decomposition_dtype,
        seed=seed,
        **compute_diffused_curvature_kwargs,
    )

    return condense_features(
        mesh,
        features,
        distance_threshold=distance_threshold,
        cluster_features=features[hks_column_names(n_components)],
    )


def compute_split_condensed_spectral(
    mesh,
    overlap_distance: float = 20_000,
    max_vertex_threshold: int = 20_000,
    min_vertex_threshold: int = 200,
    max_overlap_neighbors: int = 60_000,
    n_components: int = 32,
    n_scales: int = DEFAULT_N_SCALES,
    t_min: float = 5e4,
    t_max: float = 2e7,
    max_eigenvalue: float = 1e-5,
    robust: bool = True,
    mollify_factor: float = 1e-5,
    truncate_extra: bool = True,
    drop_first: bool = True,
    decomposition_dtype="float32",
    compute_diffused_curvature_kwargs: dict = {},
    distance_threshold: float = 3.0,
    n_jobs: Optional[int] = -1,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> tuple[pd.DataFrame, np.ndarray, MeshStitcher]:
    """Split a mesh into chunks and condense all three families on each chunk.

    The chunked middle of the composite path, shaped exactly like
    [compute_split_condensed_hks][meshmash.pipelines.condensed_hks.compute_split_condensed_hks]:
    spectral bisection into overlapping chunks, per-chunk
    [compute_condensed_spectral][meshmash.pipelines.condensed_spectral.compute_condensed_spectral],
    and reconciliation of the per-chunk domain labels into one global
    numbering.  Aggregating *within* each chunk before stitching is what keeps
    memory proportional to a chunk rather than to the whole mesh, and it is
    also why the chunking has to happen here rather than around this function:
    one eigendecomposition of the whole mesh is the thing being avoided.

    It takes the mesh as given.  Neither component thresholding nor
    simplification happens here, so a caller that has already conditioned its
    mesh gets exactly this and nothing more.

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
        Cap on overlap region size (number of nearest neighbours); overrides
        ``overlap_distance`` when set.
    n_components :
        Number of HKS timescales, for the kernel diagonal.
    n_scales :
        Number of diffusion timescales, for the curvature and tensor channels.
    t_min :
        Smallest diffusion timescale, for both grids.
    t_max :
        Largest diffusion timescale, for both grids.
    max_eigenvalue :
        Maximum Laplacian eigenvalue to decompose up to.
    robust :
        If ``True``, use the robust Laplacian (recommended).
    mollify_factor :
        Mollification factor for the robust Laplacian.
    truncate_extra :
        If ``True``, discard eigenpairs that overshoot ``max_eigenvalue``.
    drop_first :
        If ``True``, drop the constant eigenpair from the kernel diagonal.
    decomposition_dtype :
        Floating-point dtype for the eigendecomposition.
    compute_diffused_curvature_kwargs :
        Extra keyword arguments forwarded to
        [compute_diffused_curvature][meshmash.curvature.compute_diffused_curvature].
    distance_threshold :
        Ward linkage-distance threshold used to cut the agglomeration tree
        into local domains.
    n_jobs :
        Number of parallel workers for [Parallel][joblib.Parallel].
    verbose :
        Verbosity level.
    seed :
        Seed for the ARPACK starting vector.  ``None`` lets ARPACK draw its
        own, so two runs of this pipeline on the same mesh differ at
        ``decomposition_dtype`` — and Ward flips merges on those ties, so the
        domain count moves too.  An integer makes the whole pipeline
        reproducible.  Every chunk is seeded alike, which is harmless: the
        starting vector only has to overlap the wanted subspace.

    Returns
    -------
    condensed_features :
        Per-domain features indexed by global domain label, including a row
        for the null label ``-1`` whose values are all NaN.  The ``hks_``
        columns are the *log* of the area-weighted mean, matching what
        [compute_split_condensed_hks][meshmash.pipelines.condensed_hks.compute_split_condensed_hks]
        emits, so a model fit on that output reads these columns unchanged.
        The ``curvature_`` and ``normal_`` columns are the area-weighted mean
        itself.  They are not logged and cannot be: the ``normal_`` fractions
        reach zero and the ``curvature_mean_`` and ``curvature_k`` columns are
        signed.
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
        seed=seed,
    )

    if verbose:
        print("Computing spectral features across submeshes...")
    currtime = time.time()

    results_by_submesh = stitcher.apply(
        compute_condensed_spectral,
        n_components=n_components,
        n_scales=n_scales,
        t_min=t_min,
        t_max=t_max,
        max_eigenvalue=max_eigenvalue,
        robust=robust,
        mollify_factor=mollify_factor,
        truncate_extra=truncate_extra,
        drop_first=drop_first,
        decomposition_dtype=decomposition_dtype,
        compute_diffused_curvature_kwargs=compute_diffused_curvature_kwargs,
        distance_threshold=distance_threshold,
        seed=seed,
        stitch=False,
    )

    if verbose:
        print(f"Featurizing took {time.time() - currtime:.3f} seconds.")

    sub_agg_labels = stitcher.stitch_features(
        [result[1] for result in results_by_submesh],
        fill_value=-1,
    ).reshape(-1)

    agg_labels, condensed = fix_split_labels_and_features(
        sub_agg_labels,
        stitcher.submesh_mapping,
        [result[0] for result in results_by_submesh],
    )

    # The HKS block only.  See the Returns note: the other two families have
    # no log, so one np.log over the frame would replace them with NaN.
    hks_columns = hks_column_names(n_components)
    with np.errstate(divide="ignore"):
        condensed[hks_columns] = np.log(condensed[hks_columns])

    return condensed, agg_labels, stitcher
