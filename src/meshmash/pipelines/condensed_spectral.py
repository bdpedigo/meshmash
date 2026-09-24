"""Three descriptor families off one eigendecomposition per chunk, condensed.

[condensed_hks][meshmash.pipelines.condensed_hks] computes one family, the heat
kernel signature.  This computes three — the HKS, the diffused curvature
invariants, and the diffused normal-tensor invariants — from the *same*
eigendecomposition, so the second and third families cost another matrix
product per band rather than another solve.  Everything else is the HKS
pipeline's shape: a cut into overlapping chunks, per-chunk featurizing and
agglomeration, then reconciliation of the per-chunk domain numbering into one
global one.  The cut here is the geodesic Voronoi one rather than the spectral
bisection [condensed_hks][meshmash.pipelines.condensed_hks] uses: it is
cheaper, deterministic without a seed, and its chunks are connected.

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

**One key, one table.**  The condensed graph's node properties — the centroid,
the area, the vertex count — are keyed on the domain, which is the key the
spectral features already carry.  So they are a fourth column block of the same
frame, under the ``domain_`` prefix, rather than a second frame a reader has to
join back on a key it already has.  The edge table is the one thing that does
not fold in: a domain pair is a different thing from a domain.

**Conditioning is the wrapper's job, not the chunked function's.**
[compute_split_condensed_spectral][meshmash.pipelines.condensed_spectral.compute_split_condensed_spectral]
takes the mesh as given, so a caller that has already conditioned its mesh
gets that and nothing more.
[condensed_spectral_pipeline][meshmash.pipelines.condensed_spectral.condensed_spectral_pipeline]
is the layer above it that thresholds small components, simplifies, and maps
the domain labels back onto the mesh it was handed.  Its simplification
targets a vertex *density* by default rather than a reduction fraction, so
the resolution reaching the eigendecomposition does not follow the resolution
the source mesh happened to arrive at, and the timescale grid means the same
thing across meshes.
"""

import time
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..agglomerate import condense_features, fix_split_labels_and_features
from ..curvature import compute_diffused_curvature, diffused_curvature_feature_names
from ..decompose import get_hks_filter
from ..graph import condense_mesh_to_graph, condensed_node_property_names
from ..simplify import simplify_mesh, simplify_to_density
from ..split import MeshStitcher
from ..types import interpret_mesh
from ..utils import expand_labels, threshold_mesh_by_component_size

#: How many timescales the curvature and tensor channels are diffused at, when
#: the caller does not say.  Eight against the HKS's thirty-two: each of these
#: scales carries ten columns where an HKS scale carries one.
DEFAULT_N_SCALES = 8

#: Target vertex density, in vertices per unit surface area, that
#: [condensed_spectral_pipeline][meshmash.pipelines.condensed_spectral.condensed_spectral_pipeline]
#: simplifies to when the caller does not say.  A density rather than a
#: reduction fraction, so the physical resolution handed to the
#: eigendecomposition is the same whatever resolution the source mesh arrives
#: at.
DEFAULT_SIMPLIFY_TARGET_DENSITY = 4.5e-5

#: What the condensed graph's node properties are called once they sit in the
#: same frame as the spectral features.  The three spectral blocks are already
#: named for what they are, and the node properties were not named at all, so
#: they take a prefix here rather than arriving as bare ``x``, ``area``, and so
#: on next to ``hks_0``.
DOMAIN_PROPERTY_PREFIX = "domain_"


class CondensedSpectralResult(NamedTuple):
    """What
    [compute_split_condensed_spectral][meshmash.pipelines.condensed_spectral.compute_split_condensed_spectral]
    returns.

    Three grains, so three members rather than one frame: per domain, per
    domain pair, and per vertex.  The stitcher is the object that produced
    them.
    """

    #: One row per domain, keyed on the domain label.  Carries the ``hks_``,
    #: ``curvature_``, ``normal_``, and ``domain_`` blocks side by side.
    condensed_features: pd.DataFrame
    #: One row per adjacent domain pair, with ``source``, ``target``,
    #: ``boundary_length``, ``count``, and ``edge_length``.
    condensed_edges: pd.DataFrame
    #: Per-vertex domain label array of length ``V``.
    labels: np.ndarray
    #: The fitted [MeshStitcher][meshmash.split.MeshStitcher].
    stitcher: MeshStitcher


def domain_property_names(add_component_features: bool = True) -> list[str]:
    """The ``domain_`` columns of a condensed spectral table, in order.

    The condensed graph's node properties under
    [DOMAIN_PROPERTY_PREFIX][meshmash.pipelines.condensed_spectral.DOMAIN_PROPERTY_PREFIX].
    Derived from
    [condensed_node_property_names][meshmash.graph.condensed_node_property_names]
    rather than rebuilt, so the two cannot drift apart.

    Parameters
    ----------
    add_component_features :
        Whether the component columns are included, matching the argument of
        the same name on
        [condense_mesh_to_graph][meshmash.graph.condense_mesh_to_graph].

    Returns
    -------
    :
        ``["domain_x", "domain_y", "domain_z", ...]``.
    """
    return [
        DOMAIN_PROPERTY_PREFIX + name
        for name in condensed_node_property_names(add_component_features)
    ]


def condensed_spectral_column_names(n_components: int, n_scales: int) -> list[str]:
    """Every column of a condensed spectral table, in order.

    The four blocks a caller declaring a schema has to know about: the three
    spectral families from
    [diffused_curvature_feature_names][meshmash.curvature.diffused_curvature_feature_names],
    then the condensed graph's node properties from
    [domain_property_names][meshmash.pipelines.condensed_spectral.domain_property_names].
    The column set depends on the parameters, which is why this is a function
    of them rather than a constant.

    Parameters
    ----------
    n_components :
        Number of HKS timescales.
    n_scales :
        Number of diffusion timescales.

    Returns
    -------
    :
        Column names, of length ``n_components + 6 + n_scales * 10 + 7``.
    """
    return (
        diffused_curvature_feature_names(n_scales, n_diagonal=n_components)
        + domain_property_names()
    )


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
    blas_threads: Optional[int] = 1,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Featurize and condense a single (unsplit) mesh, three families at once.

    The counterpart of
    [compute_condensed_hks][meshmash.pipelines.condensed_hks.compute_condensed_hks]:
    a lightweight helper that
    [compute_split_condensed_spectral][meshmash.pipelines.condensed_spectral.compute_split_condensed_spectral]
    runs on each chunk.  For a large mesh, use that instead.

    No ``domain_`` block comes out of here, unlike the chunked function that
    calls it.  Its domains are chunk-local and its chunks overlap, so an area
    or a vertex count measured on one chunk counts the overlap region again on
    the next.  Those properties are only well defined once the labels are one
    global numbering over the whole mesh.

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
    blas_threads :
        BLAS thread count for the linear algebra, fixed rather than inherited.
        The reduction order depends on it, which moves the features in their
        last float32 bit, and Ward flips a near-tie on that: measured on a
        jittered sphere the domain count ranges over 527 to 563 purely with
        worker count, because joblib gives its workers a different thread count
        than an in-process call. Any fixed value is reproducible, so this is
        ``1`` rather than tuned. ``None`` inherits the ambient count and gives
        up reproducibility across worker counts.

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
    # The linear algebra below sums in an order that depends on how many BLAS
    # threads it gets, which changes the features in their last float32 bit.
    # Ward is greedy, so a near-tie flips and the domains move: measured on a
    # jittered sphere, the domain count ranges over 527 to 563 across worker
    # counts, because joblib gives its workers a different thread count than an
    # in-process call. Fixing the thread count makes the whole pipeline
    # reproducible whatever `n_jobs` is. See TASK-12.
    with threadpool_limits(limits=blas_threads):
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
    target_vertices: int = 10_000,
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
    blas_threads: Optional[int] = 1,
) -> CondensedSpectralResult:
    """Split a mesh into chunks and condense all three families on each chunk.

    The chunked middle of the composite path, shaped like
    [compute_split_condensed_hks][meshmash.pipelines.condensed_hks.compute_split_condensed_hks]:
    a cut into overlapping chunks, per-chunk
    [compute_condensed_spectral][meshmash.pipelines.condensed_spectral.compute_condensed_spectral],
    and reconciliation of the per-chunk domain labels into one global
    numbering.  The cut is
    [fit_mesh_split_geodesic][meshmash.split.fit_mesh_split_geodesic], not the
    spectral bisection, so the chunk boundaries depend on the mesh alone and
    every chunk comes out connected.  Aggregating *within* each chunk before stitching is what keeps
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
    target_vertices :
        Vertices to aim for in each core chunk, which sets how many Voronoi
        seeds a piece is cut with.  Chunks come out near this size and always
        under ``max_vertex_threshold``.
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
        Seed for the ARPACK starting vector of the featurizing
        eigendecomposition.  The chunking does not read it: the geodesic cut is
        deterministic on its own.  ``None`` lets ARPACK draw its own vector, so
        two runs of this pipeline on the same mesh differ at
        ``decomposition_dtype`` — and Ward flips merges on those ties, so the
        domain count moves too.  An integer makes the whole pipeline
        reproducible.  Every chunk is seeded alike, which is harmless: the
        starting vector only has to overlap the wanted subspace.
    blas_threads :
        BLAS thread count for the linear algebra, fixed rather than inherited.
        The reduction order depends on it, which moves the features in their
        last float32 bit, and Ward flips a near-tie on that: measured on a
        jittered sphere the domain count ranges over 527 to 563 purely with
        worker count, because joblib gives its workers a different thread count
        than an in-process call. Any fixed value is reproducible, so this is
        ``1`` rather than tuned. ``None`` inherits the ambient count and gives
        up reproducibility across worker counts.

    Returns
    -------
    :
        A
        [CondensedSpectralResult][meshmash.pipelines.condensed_spectral.CondensedSpectralResult]
        of ``condensed_features``, ``condensed_edges``, ``labels``, and
        ``stitcher``.

        ``condensed_features`` is indexed by global domain label and includes
        a row for the null label ``-1``, whose values are all NaN.  The
        ``hks_`` columns are the *log* of the area-weighted mean, matching
        what
        [compute_split_condensed_hks][meshmash.pipelines.condensed_hks.compute_split_condensed_hks]
        emits, so a model fit on that output reads these columns unchanged.
        The ``curvature_`` and ``normal_`` columns are the area-weighted mean
        itself.  They are not logged and cannot be: the ``normal_`` fractions
        reach zero and the ``curvature_mean_`` and ``curvature_k`` columns are
        signed.  The ``domain_`` columns are the condensed graph's node
        properties, which are sums and centroids rather than means.

        ``condensed_edges`` is one row per adjacent domain pair.  It stays a
        separate table because a pair is a different thing from a domain, so
        it has a different key and a different row count.

        ``labels`` is a per-vertex array of length ``V``, ``-1`` for a vertex
        in no domain.
    """
    stitcher = MeshStitcher(mesh, n_jobs=n_jobs, verbose=verbose)
    stitcher.split_mesh(
        method="geodesic",
        overlap_distance=overlap_distance,
        max_vertex_threshold=max_vertex_threshold,
        min_vertex_threshold=min_vertex_threshold,
        max_overlap_neighbors=max_overlap_neighbors,
        target_vertices=target_vertices,
        verify_connected=False,
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
        blas_threads=blas_threads,
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

    if verbose:
        print("Condensing the mesh to a domain graph...")
    currtime = time.time()

    # `stitcher.mesh`, not `mesh`: the argument may be anything
    # `interpret_mesh` accepts, and the labels index the interpreted arrays.
    condensed_nodes, condensed_edges = condense_mesh_to_graph(
        stitcher.mesh, agg_labels, add_component_features=True
    )

    if verbose:
        print(f"Condensing took {time.time() - currtime:.3f} seconds.")

    # The node table has no row for the null label and the feature table does,
    # so reindexing onto the feature index is what fills that row with NaN and
    # keeps the two blocks in one order.  The counts join as float32, which is
    # exact to 2**24 vertices, so the frame stays one float32 matrix with a NaN
    # row rather than mixing in a nullable integer column.
    condensed_nodes = condensed_nodes.rename(
        columns=lambda name: DOMAIN_PROPERTY_PREFIX + name
    )[domain_property_names()].astype(np.float32)
    condensed = condensed.join(condensed_nodes.reindex(condensed.index))

    return CondensedSpectralResult(condensed, condensed_edges, agg_labels, stitcher)


class CondensedSpectralPipelineResult(NamedTuple):
    """What
    [condensed_spectral_pipeline][meshmash.pipelines.condensed_spectral.condensed_spectral_pipeline]
    returns.

    Wider than
    [CondensedSpectralResult][meshmash.pipelines.condensed_spectral.CondensedSpectralResult]
    by the conditioning the pipeline does: the mesh the featurizing actually
    ran on, the map from the input mesh onto it, and labels at both
    resolutions.  Shaped to match
    [CondensedHKSResult][meshmash.pipelines.condensed_hks.CondensedHKSResult],
    minus its ``condensed_nodes`` member, which is the ``domain_`` block of
    ``condensed_features`` here.
    """

    #: The thresholded and simplified ``(vertices, faces)`` tuple the
    #: featurizing ran on.
    simple_mesh: tuple
    #: Array of length ``V_original`` giving each input vertex its index in
    #: ``simple_mesh``.  ``-1`` for a vertex dropped by the component
    #: threshold.
    mapping: np.ndarray
    #: The fitted [MeshStitcher][meshmash.split.MeshStitcher], over
    #: ``simple_mesh``.
    stitcher: MeshStitcher
    #: Per-vertex domain label array over ``simple_mesh``.
    simple_labels: np.ndarray
    #: Per-vertex domain label array over the *input* mesh, of length
    #: ``V_original``.
    labels: np.ndarray
    #: One row per domain, keyed on the domain label.  Carries the ``hks_``,
    #: ``curvature_``, ``normal_``, and ``domain_`` blocks side by side, the
    #: ``domain_`` block measured on the input mesh.
    condensed_features: pd.DataFrame
    #: One row per adjacent domain pair, measured on the input mesh.
    condensed_edges: pd.DataFrame
    #: Wall-clock seconds per pipeline step.
    timing_info: dict


def condensed_spectral_pipeline(
    mesh,
    simplify_agg: int = 7,
    simplify_target_reduction: Optional[float] = None,
    simplify_target_density: Optional[float] = DEFAULT_SIMPLIFY_TARGET_DENSITY,
    overlap_distance: float = 20_000,
    max_vertex_threshold: int = 20_000,
    min_vertex_threshold: int = 200,
    max_overlap_neighbors: int = 60_000,
    target_vertices: int = 10_000,
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
    blas_threads: Optional[int] = 1,
) -> CondensedSpectralPipelineResult:
    """Condition a mesh, then featurize and condense it, three families at once.

    The entry point for the composite spectral path, and the counterpart of
    [condensed_hks_pipeline][meshmash.pipelines.condensed_hks.condensed_hks_pipeline]:
    component thresholding and simplification, then
    [compute_split_condensed_spectral][meshmash.pipelines.condensed_spectral.compute_split_condensed_spectral]
    on what comes out, then the domain labels expanded back onto the input
    mesh.  Call
    [compute_split_condensed_spectral][meshmash.pipelines.condensed_spectral.compute_split_condensed_spectral]
    directly for a mesh the caller has already conditioned.

    Simplification targets a vertex *density* by default, where
    [condensed_hks_pipeline][meshmash.pipelines.condensed_hks.condensed_hks_pipeline]
    targets a reduction fraction.  The two knobs are mutually exclusive: to
    target a reduction fraction here, pass ``simplify_target_density=None``.
    Passing ``None`` to both skips simplification.

    The ``domain_`` block and the edge table are measured on the *input* mesh
    under the expanded labels, not on the simplified mesh the features came
    from.


    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    simplify_agg :
        Decimation aggressiveness (0-10).  Higher values are faster but
        reduce mesh quality.
    simplify_target_reduction :
        Fraction of triangles to remove, for
        [simplify_mesh][meshmash.simplify.simplify_mesh].  Mutually exclusive
        with ``simplify_target_density``.
    simplify_target_density :
        Target vertex density (vertices per unit surface area) for
        [simplify_to_density][meshmash.simplify.simplify_to_density].
        Defaults to
        [DEFAULT_SIMPLIFY_TARGET_DENSITY][meshmash.pipelines.condensed_spectral.DEFAULT_SIMPLIFY_TARGET_DENSITY].
        Mutually exclusive with ``simplify_target_reduction``; providing both
        raises ``ValueError``.
    overlap_distance :
        Geodesic radius used to grow each chunk into its overlap region.
    max_vertex_threshold :
        Maximum vertices per core chunk before overlapping.
    min_vertex_threshold :
        Minimum connected-component size.  Smaller components are removed
        from the mesh before simplification, and their vertices come back
        with label ``-1``.
    max_overlap_neighbors :
        Cap on overlap region size (number of nearest neighbours); overrides
        ``overlap_distance`` when set.
    target_vertices :
        Vertices to aim for in each core chunk.
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
        Seed for the ARPACK starting vector of the featurizing
        eigendecomposition.  ``None`` gives up reproducibility, for the
        reason
        [compute_split_condensed_spectral][meshmash.pipelines.condensed_spectral.compute_split_condensed_spectral]
        records.  An integer is sufficient for a reproducible run: the
        simplifier takes no seed and is deterministic on its own.
    blas_threads :
        BLAS thread count for the linear algebra, fixed rather than
        inherited.  ``None`` inherits the ambient count and gives up
        reproducibility across worker counts.

    Returns
    -------
    :
        A
        [CondensedSpectralPipelineResult][meshmash.pipelines.condensed_spectral.CondensedSpectralPipelineResult].
        Its ``condensed_features`` carries the columns
        [condensed_spectral_column_names][meshmash.pipelines.condensed_spectral.condensed_spectral_column_names]
        gives, indexed by global domain label and including a NaN row for the
        null label ``-1``.

    Raises
    ------
    ValueError
        If both ``simplify_target_reduction`` and ``simplify_target_density``
        are given.
    """
    if simplify_target_reduction is not None and simplify_target_density is not None:
        raise ValueError(
            "Provide only one of `simplify_target_reduction` or "
            "`simplify_target_density`, not both. This pipeline targets a "
            "density by default, so to use reduction-targeted simplification, "
            "set `simplify_target_density=None`."
        )

    timing_info = {}
    starttime = time.time()

    original_mesh = interpret_mesh(mesh)

    currtime = time.time()
    thresholded_mesh, indices_from_original = threshold_mesh_by_component_size(
        original_mesh, size_threshold=min_vertex_threshold
    )

    if simplify_target_density is not None:
        vertices, faces, thresh_to_simple_mapping = simplify_to_density(
            thresholded_mesh,
            target_density=simplify_target_density,
            simplify_agg=simplify_agg,
            verbose=verbose,
        )
        simple_mesh = (vertices, faces)
    else:
        # `simplify_mesh` also covers `target_reduction=None`, which returns
        # the mesh untouched with an identity mapping.
        simple_mesh, thresh_to_simple_mapping = simplify_mesh(
            thresholded_mesh,
            agg=simplify_agg,
            target_reduction=simplify_target_reduction,
        )
    timing_info["conditioning_time"] = time.time() - currtime

    currtime = time.time()
    result = compute_split_condensed_spectral(
        simple_mesh,
        overlap_distance=overlap_distance,
        max_vertex_threshold=max_vertex_threshold,
        min_vertex_threshold=min_vertex_threshold,
        max_overlap_neighbors=max_overlap_neighbors,
        target_vertices=target_vertices,
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
        n_jobs=n_jobs,
        verbose=verbose,
        seed=seed,
        blas_threads=blas_threads,
    )
    timing_info["spectral_time"] = time.time() - currtime

    mapping = np.full(len(original_mesh[0]), -1, dtype=np.int32)
    mapping[indices_from_original] = thresh_to_simple_mapping

    labels = expand_labels(result.labels, mapping)

    if verbose:
        print("Condensing the mesh to a domain graph...")
    currtime = time.time()

    # Recomputed on `original_mesh`, replacing the block
    # `compute_split_condensed_spectral` measured on the simplified mesh. An
    # area or a vertex count is a property of the mesh it is measured on, and
    # simplification changes both, so the numbers a caller wants are the ones
    # the input mesh gives.
    condensed_nodes, condensed_edges = condense_mesh_to_graph(
        original_mesh, labels, add_component_features=True
    )
    condensed_nodes = condensed_nodes.rename(
        columns=lambda name: DOMAIN_PROPERTY_PREFIX + name
    )[domain_property_names()].astype(np.float32)
    condensed = result.condensed_features.drop(columns=domain_property_names())
    condensed = condensed.join(condensed_nodes.reindex(condensed.index))

    timing_info["condense_time"] = time.time() - currtime
    timing_info["pipeline_time"] = time.time() - starttime

    return CondensedSpectralPipelineResult(
        simple_mesh,
        mapping,
        result.stitcher,
        result.labels,
        labels,
        condensed,
        condensed_edges,
        timing_info,
    )
