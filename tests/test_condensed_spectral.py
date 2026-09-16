"""The composite pipeline: three families, one eigendecomposition, per chunk.

The band-by-band solve starts ARPACK from a random vector, so two runs of the
same call agree only to the decomposition dtype — float32 here. Nothing below
compares two separate featurizations for equality, and the one test that
compares two of them at all carries a tolerance that says so.
"""

import numpy as np
import pandas as pd
import pytest
import pyvista as pv

from meshmash import (
    compute_diffused_curvature,
    compute_hks,
    compute_split_condensed_hks,
    compute_split_condensed_spectral,
    condense_features,
    cotangent_laplacian,
    diffused_curvature_feature_names,
)
from meshmash.decompose import get_hks_filter
from meshmash.pipelines.condensed_spectral import (
    DEFAULT_SIMPLIFY_TARGET_DENSITY,
    compute_condensed_spectral,
    condensed_spectral_column_names,
    condensed_spectral_pipeline,
    domain_property_names,
    hks_column_names,
)
from meshmash.utils import poly_to_mesh, vertex_density

N_COMPONENTS = 4
N_SCALES = 3
MAX_EIGENVALUE = 1e-4
SCALES = np.geomspace(1e4, 2.5e5, N_SCALES)

#: Passed to both featurizers wherever they are compared, because their
#: defaults disagree: `compute_hks` truncates and drops nothing, while
#: `compute_diffused_curvature` does both (TASK-9.1). The pipelines pass these
#: explicitly for the same reason, so this is the configuration under test.
SPECTRUM_FLAGS = dict(truncate_extra=True, drop_first=True)


@pytest.fixture(scope="module")
def sphere():
    """Jittered, because a round sphere cannot be compared against itself.

    A round sphere's eigenvalues have multiplicity 2l+1, and the band-by-band
    solve cuts a degenerate eigenspace wherever its random start vector lands.
    Two runs then truncate a different set of modes, which moves the kernel
    diagonal by tens of percent rather than by rounding.
    """
    poly = pv.Sphere(radius=1000.0, theta_resolution=24, phi_resolution=24)
    vertices, faces = poly_to_mesh(poly.triangulate())
    rng = np.random.default_rng(3)
    scaling = 1.0 + 0.05 * rng.normal(size=(len(vertices), 1))
    return (np.asarray(vertices) * scaling, np.asarray(faces))


@pytest.fixture(scope="module")
def sphere_features(sphere):
    """One featurization, reused: a second call would differ at float32."""
    return compute_diffused_curvature(
        sphere,
        SCALES,
        diagonal_filter=get_hks_filter(
            SCALES[-1], SCALES[0], N_COMPONENTS, dtype=np.float32
        ),
        max_eigenvalue=MAX_EIGENVALUE,
        decomposition_dtype=np.float32,
        **SPECTRUM_FLAGS,
    )


@pytest.fixture(scope="module")
def condensed(mesh):
    """One run of the whole chunked pipeline on the sample dendrite."""
    return compute_split_condensed_spectral(
        mesh,
        n_components=N_COMPONENTS,
        n_scales=N_SCALES,
        max_eigenvalue=1e-8,
        max_vertex_threshold=5000,
        n_jobs=1,
        verbose=False,
    )


# --- cutting on one block, aggregating every block ------------------------


def test_cluster_features_moves_the_cut_and_not_the_columns(sphere, sphere_features):
    """The mechanism the composite is built on.

    Cutting on the HKS block alone has to give the domains that block alone
    would give, while the aggregated table still carries all three families.
    """
    hks_block = sphere_features[hks_column_names(N_COMPONENTS)]

    both, labels = condense_features(
        sphere, sphere_features, cluster_features=hks_block
    )
    _, hks_only_labels = condense_features(sphere, hks_block)

    np.testing.assert_array_equal(labels, hks_only_labels)
    assert list(both.columns) == list(sphere_features.columns)


def test_cluster_features_defaults_to_the_aggregated_features(sphere, sphere_features):
    """Passing the same block both ways is the single-family call."""
    hks_block = sphere_features[hks_column_names(N_COMPONENTS)]

    explicit, explicit_labels = condense_features(
        sphere, hks_block, cluster_features=hks_block
    )
    implied, implied_labels = condense_features(sphere, hks_block)

    np.testing.assert_array_equal(explicit_labels, implied_labels)
    pd.testing.assert_frame_equal(explicit, implied)


def test_cluster_features_rejects_a_mismatched_length(sphere, sphere_features):
    with pytest.raises(ValueError, match="per-vertex"):
        condense_features(
            sphere, sphere_features, cluster_features=sphere_features.to_numpy()[:-1]
        )


# --- one decomposition, three families ------------------------------------


def test_the_fused_diagonal_is_the_heat_kernel_signature(sphere, sphere_features):
    """What makes the second and third families free.

    If the diagonal half of the concatenated bank were not the HKS, the
    composite would be computing a different first family than the pipeline it
    is meant to stay comparable with. The tolerance is the float32
    decomposition's, not the method's: two band-by-band solves of the same
    operator start from different random vectors.
    """
    expected = compute_hks(
        sphere,
        t_min=SCALES[0],
        t_max=SCALES[-1],
        n_components=N_COMPONENTS,
        max_eigenvalue=MAX_EIGENVALUE,
        decomposition_dtype=np.float32,
        **SPECTRUM_FLAGS,
    )
    diagonal = sphere_features[hks_column_names(N_COMPONENTS)].to_numpy()

    np.testing.assert_allclose(diagonal, expected, rtol=1e-4)


def test_the_two_grids_are_independent(sphere):
    """`n_scales` buys more curvature columns and leaves the HKS block alone."""
    coarse = compute_condensed_spectral(
        sphere, n_components=N_COMPONENTS, n_scales=2, max_eigenvalue=MAX_EIGENVALUE
    )[0]
    fine = compute_condensed_spectral(
        sphere, n_components=N_COMPONENTS, n_scales=6, max_eigenvalue=MAX_EIGENVALUE
    )[0]

    assert list(coarse.columns) == diffused_curvature_feature_names(2, N_COMPONENTS)
    assert list(fine.columns) == diffused_curvature_feature_names(6, N_COMPONENTS)
    assert hks_column_names(N_COMPONENTS) == list(coarse.columns[:N_COMPONENTS])


# --- the chunked pipeline -------------------------------------------------


def test_the_pipeline_returns_one_named_table_per_domain(mesh, condensed):
    """One frame, four blocks, and a null row that is NaN across all of them."""
    features, labels, stitcher = (
        condensed.condensed_features,
        condensed.labels,
        condensed.stitcher,
    )

    assert list(features.columns) == condensed_spectral_column_names(
        N_COMPONENTS, N_SCALES
    )
    assert list(features.columns[: -len(domain_property_names())]) == (
        diffused_curvature_feature_names(N_SCALES, N_COMPONENTS)
    )
    assert len(stitcher.submeshes) > 1, "the point is that it ran on chunks"
    assert len(labels) == len(mesh[0])
    assert labels.max() == features.index.max()
    assert list(features.index) == [-1] + list(range(labels.max() + 1))
    assert features.loc[-1].isna().all()


def test_every_domain_has_finite_features(condensed):
    """A domain no chunk could featurize would be NaN across the board."""
    features = condensed.condensed_features.drop(index=-1)

    assert np.isfinite(features.to_numpy()).all()


def test_only_the_hks_block_is_logged(condensed):
    """The families are aggregated on different terms, and that has to show.

    The shape fractions sum to one per vertex, and an area-weighted mean of
    values summing to one still sums to one. A log anywhere in that block
    would destroy it. The curvature columns are the other direction: a
    dendrite has saddles, so Gaussian curvature and the smaller principal
    curvature come out negative on some domains, and a logged column could not
    hold both signs. Mean curvature is not the column to check — a dendrite is
    convex on average, so its domain means are all positive either way.
    """
    features = condensed.condensed_features.drop(index=-1)

    for scale in range(N_SCALES):
        fractions = features[
            [f"normal_{name}_{scale}" for name in ("sheet", "tube", "blob")]
        ]
        np.testing.assert_allclose(fractions.sum(axis=1), 1.0, atol=1e-6)

    for column in ("curvature_gauss_raw", "curvature_k2_0"):
        values = features[column]
        assert (values > 0).any() and (values < 0).any(), column


def test_the_hks_block_is_logged(condensed):
    """The half of the same claim that keeps the columns drop-in comparable.

    `compute_split_condensed_hks` logs what it emits, so anything reading its
    output reads a log. These HKS values sit near 1e-8, whose log is about
    -18, and no unlogged HKS is negative.
    """
    hks = condensed.condensed_features.drop(index=-1)[hks_column_names(N_COMPONENTS)]

    assert (hks < 0).all().all()


# --- the folded node-property block ---------------------------------------


def test_the_domain_block_measures_the_domains_the_labels_name(mesh, condensed):
    """The fold is only sound if the two blocks describe the same domains.

    The node properties are sums over the vertices of a domain, so they are
    checkable against the label array directly: the vertex counts have to be
    the counts of each label, and they have to add up to the number of
    labeled vertices. A block joined onto the wrong index would fail both.
    """
    features = condensed.condensed_features
    labels = condensed.labels
    counts = features.loc[features.index != -1, "domain_n_vertices"]

    expected = pd.Series(labels[labels != -1]).value_counts().sort_index()
    np.testing.assert_array_equal(counts.to_numpy(), expected.to_numpy())
    assert counts.sum() == (labels != -1).sum()


def test_the_domain_block_is_finite_and_null_only_on_the_null_label(condensed):
    """`condense_mesh_to_graph` emits no row for -1, so the join has to make one."""
    features = condensed.condensed_features
    block = features[domain_property_names()]

    assert block.loc[-1].isna().all()
    assert np.isfinite(block.drop(index=-1).to_numpy()).all()


def test_the_component_columns_are_constant_within_a_component(condensed):
    """A component property repeated on every domain of that component.

    The dendrite sample is one component, so every domain carries the same
    pair, and that pair is the total over the labeled vertices.
    """
    features = condensed.condensed_features.drop(index=-1)

    assert features["domain_component_n_vertices"].nunique() == 1
    assert (
        features["domain_component_n_vertices"].iloc[0]
        == features["domain_n_vertices"].sum()
    )
    np.testing.assert_allclose(
        features["domain_component_area"].iloc[0],
        features["domain_area"].sum(),
        rtol=1e-6,
    )


def test_the_edges_keep_their_own_grain(condensed):
    """A domain pair is not a domain, so it stays a second table.

    Every endpoint has to be a real domain, no pair may repeat, and no domain
    may be paired with itself. Those three are what make the pair the key.
    """
    edges = condensed.condensed_edges
    features = condensed.condensed_features

    assert list(edges.columns) == [
        "source",
        "target",
        "boundary_length",
        "count",
        "edge_length",
    ]
    assert edges["source"].isin(features.index).all()
    assert edges["target"].isin(features.index).all()
    assert (edges["source"] < edges["target"]).all()
    assert not edges.duplicated(subset=["source", "target"]).any()
    assert len(edges) != len(features), "the two grains would be confusable"


def test_the_domain_prefix_keeps_the_blocks_apart(condensed):
    """The prefix is the whole reason the four blocks can share a frame."""
    columns = list(condensed.condensed_features.columns)

    prefixed = [name for name in columns if name.startswith("domain_")]
    assert prefixed == domain_property_names()
    assert not any(
        name.startswith(("hks_", "curvature_", "normal_")) for name in prefixed
    )


# --- reproducibility and parity with the HKS pipeline ---------------------


def test_a_seed_makes_the_pipeline_reproducible(mesh):
    """Without one, ARPACK draws its own start vector and Ward flips merges.

    The variation is not in the method. ARPACK's generator carries state
    across calls inside a process, so the second run of a call starts
    somewhere else, and at float32 the resulting HKS differ by about 1e-6 —
    enough for connectivity-constrained Ward to merge a near-tie the other
    way and return a different number of domains.

    The chunking does not need the seed: the geodesic cut this pipeline makes
    is deterministic on its own, so only the solves vary between runs.
    """
    kwargs = dict(
        n_components=N_COMPONENTS,
        n_scales=N_SCALES,
        max_eigenvalue=1e-8,
        max_vertex_threshold=5000,
        n_jobs=1,
        seed=0,
    )
    first = compute_split_condensed_spectral(mesh, **kwargs)
    second = compute_split_condensed_spectral(mesh, **kwargs)

    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(
        first.condensed_features.to_numpy(), second.condensed_features.to_numpy()
    )
    pd.testing.assert_frame_equal(first.condensed_edges, second.condensed_edges)


def test_the_domain_numbering_does_not_follow_worker_count(mesh):
    """Domain labels must name the same domains at any ``n_jobs`` (TASK-12).

    The label is the join key between this table and everything derived from
    the same cut, so a label that moves with worker count cannot be committed.
    Before `canonicalize_labels`, global labels were handed out in
    ``(submesh index, local label)`` order, and the local part is not stable:
    joblib caps its workers' BLAS threads where an in-process run uses every
    one, and the different reduction order flips Ward's near-ties. The
    partition that survives overlap trimming was the same, so only the names
    moved.

    This does not assert the features, because a weighted mean accumulates in
    a different order per worker count and the columns differ by about 1e-6 at
    float32. It also stops at two worker counts on purpose: at higher ones the
    *partition* itself moves, which renumbering cannot fix and TASK-12 records
    as still open.
    """
    kwargs = dict(
        n_components=N_COMPONENTS,
        n_scales=N_SCALES,
        max_eigenvalue=1e-8,
        max_vertex_threshold=5000,
        seed=0,
    )
    serial = compute_split_condensed_spectral(mesh, n_jobs=1, **kwargs)
    parallel = compute_split_condensed_spectral(mesh, n_jobs=2, **kwargs)

    np.testing.assert_array_equal(serial.labels, parallel.labels)


def test_domain_zero_holds_the_lowest_numbered_vertex(mesh):
    """The canonical numbering rule, stated as a property.

    Labels run in order of first appearance along the vertex array, so the
    numbering is a function of the partition and of nothing else. Asserting the
    rule rather than a recorded label array means the test still means
    something if the cut changes.
    """
    labels = compute_split_condensed_spectral(
        mesh,
        n_components=N_COMPONENTS,
        n_scales=N_SCALES,
        max_eigenvalue=1e-8,
        max_vertex_threshold=5000,
        n_jobs=1,
        seed=0,
    ).labels
    assigned = labels[labels != -1]
    first_appearance = pd.unique(assigned)

    np.testing.assert_array_equal(first_appearance, np.arange(len(first_appearance)))


def test_the_composite_reproduces_the_hks_pipeline_exactly(mesh):
    """Adding two families must not move the domains the HKS alone would find.

    Both sides are put on the geodesic cut, which is what the composite
    pipeline uses and what the HKS pipeline takes as an option. The claim is
    about the featurizing, so the chunking has to be held fixed: on its default
    spectral cut the HKS pipeline chunks the mesh somewhere else and the domains
    move with the chunk boundaries, for a reason that has nothing to do with the
    extra families.

    On one cut, and with the solves seeded, the two pipelines put every vertex
    in the same domain at either dtype. The per-vertex HKS underneath is
    bit-identical: the fused bank's diagonal half is the same arithmetic
    `compute_hks` does, on the same operator.

    The aggregated values are bit-identical only at float64. At float32 they
    differ by about 1e-7, which is that dtype's epsilon: the area-weighted mean
    runs over 40 columns here and 4 there, and a weighted sum rounds by the
    order it accumulates in. Nothing about the method differs, so at float32
    only the domains are asserted.
    """
    kwargs = dict(
        n_components=N_COMPONENTS,
        max_eigenvalue=1e-8,
        max_vertex_threshold=5000,
        target_vertices=2500,
        n_jobs=1,
        seed=0,
    )

    for dtype, values_agree in (("float32", False), ("float64", True)):
        composite = compute_split_condensed_spectral(
            mesh, n_scales=N_SCALES, decomposition_dtype=dtype, **kwargs
        )
        hks_only, hks_labels, _ = compute_split_condensed_hks(
            mesh, decomposition_dtype=dtype, method="geodesic", **kwargs
        )

        assert len(composite.stitcher.submeshes) > 1, (
            "the point is that it ran on chunks"
        )
        np.testing.assert_array_equal(composite.labels, hks_labels, err_msg=dtype)

        diagonal = composite.condensed_features[
            hks_column_names(N_COMPONENTS)
        ].to_numpy()
        if values_agree:
            np.testing.assert_array_equal(diagonal, hks_only.to_numpy())
        else:
            np.testing.assert_allclose(diagonal, hks_only.to_numpy(), rtol=1e-6)


def test_dropping_the_constant_mode_removes_one_number(sphere):
    """What `drop_first` actually does, against what its old docstring claimed.

    The constant eigenpair contributes ``1 / total_area`` to the kernel
    diagonal at every vertex and every timescale, so dropping it subtracts one
    number everywhere. It is not a per-vertex area normalization: the vertex
    areas of this sphere span more than an order of magnitude, and none of
    that spread appears in the difference.
    """
    _, M = cotangent_laplacian(sphere, robust=True, mollify_factor=1e-5)
    areas = np.asarray(M.diagonal())
    kwargs = dict(
        t_min=SCALES[0],
        t_max=SCALES[-1],
        n_components=N_COMPONENTS,
        max_eigenvalue=MAX_EIGENVALUE,
        truncate_extra=True,
        seed=0,
    )

    kept = compute_hks(sphere, drop_first=False, **kwargs)
    dropped = compute_hks(sphere, drop_first=True, **kwargs)

    assert areas.max() / areas.min() > 10, "a flat sphere would prove nothing"
    np.testing.assert_allclose(kept - dropped, 1.0 / areas.sum(), rtol=1e-9)


# --- the conditioning wrapper ---------------------------------------------

#: What the wrapper is run with wherever it is run below. Small enough to be
#: quick, chunked enough that the stitching is exercised.
PIPELINE_KWARGS = dict(
    n_components=N_COMPONENTS,
    n_scales=N_SCALES,
    max_eigenvalue=1e-8,
    max_vertex_threshold=5000,
    n_jobs=1,
    seed=0,
)


@pytest.fixture(scope="module")
def pipelined(mesh):
    """One run of the conditioning wrapper at its density default."""
    return condensed_spectral_pipeline(mesh, **PIPELINE_KWARGS)


def test_the_pipeline_simplifies_to_the_default_density(mesh, pipelined):
    """The default is a density, so the output density is the thing to check.

    `simplify_to_density` stops within 5% above the target, so the assertion
    is one-sided on that tolerance rather than a two-sided closeness.
    """
    assert vertex_density(mesh) > DEFAULT_SIMPLIFY_TARGET_DENSITY, (
        "a mesh already below the target would not be simplified at all"
    )
    density = vertex_density(pipelined.simple_mesh)
    assert density <= DEFAULT_SIMPLIFY_TARGET_DENSITY * 1.05
    assert len(pipelined.simple_mesh[0]) < len(mesh[0])


def test_the_two_simplification_knobs_are_mutually_exclusive(mesh):
    """Density is the default here, so a bare reduction argument gives both."""
    with pytest.raises(ValueError, match="only one of"):
        condensed_spectral_pipeline(mesh, simplify_target_reduction=0.7)


def test_the_pipeline_takes_a_reduction_fraction_instead(mesh):
    """The other branch, and the one that skips simplification entirely."""
    reduced = condensed_spectral_pipeline(
        mesh,
        simplify_target_density=None,
        simplify_target_reduction=0.7,
        **PIPELINE_KWARGS,
    )
    untouched = condensed_spectral_pipeline(
        mesh,
        simplify_target_density=None,
        simplify_target_reduction=None,
        **PIPELINE_KWARGS,
    )

    assert len(reduced.simple_mesh[0]) < len(mesh[0])
    assert len(untouched.simple_mesh[0]) == len(mesh[0])
    np.testing.assert_array_equal(untouched.mapping, np.arange(len(mesh[0])))


def test_the_pipeline_labels_the_mesh_it_was_handed(mesh, pipelined):
    """Two label arrays at two resolutions, and the map between them."""
    assert len(pipelined.labels) == len(mesh[0])
    assert len(pipelined.simple_labels) == len(pipelined.simple_mesh[0])
    assert len(pipelined.mapping) == len(mesh[0])

    kept = pipelined.mapping != -1
    np.testing.assert_array_equal(
        pipelined.labels[kept], pipelined.simple_labels[pipelined.mapping[kept]]
    )
    assert (pipelined.labels[~kept] == -1).all()


def test_the_domain_block_is_measured_on_the_input_mesh(mesh, pipelined):
    """Not on the simplified mesh the features came from.

    An area or a vertex count is a property of the mesh it is measured on,
    and simplification moves both. The counts have to add up against the
    labels over the *input* mesh.
    """
    features = pipelined.condensed_features
    real = features.index[features.index != -1]

    labels, counts = np.unique(pipelined.labels, return_counts=True)
    # Reindexed rather than `.drop(-1)`: on a mesh whose components all clear
    # the threshold there is no unlabeled vertex, so there is no -1 to drop.
    expected = pd.Series(counts, index=labels).reindex(real)

    np.testing.assert_array_equal(
        features.loc[real, "domain_n_vertices"].to_numpy(),
        expected.to_numpy(),
    )
    assert (
        features.loc[real, "domain_n_vertices"].sum() == (pipelined.labels != -1).sum()
    )
    assert features.loc[real, "domain_n_vertices"].sum() > len(
        pipelined.simple_mesh[0]
    ), "the simplified mesh has fewer vertices, so this would fail if measured there"


def test_the_pipeline_keeps_the_column_contract(pipelined):
    """Same four blocks in the same order as the unconditioned function."""
    assert list(
        pipelined.condensed_features.columns
    ) == condensed_spectral_column_names(N_COMPONENTS, N_SCALES)
    assert -1 in pipelined.condensed_features.index
    assert pipelined.condensed_features.loc[-1, domain_property_names()].isna().all()
    assert list(pipelined.condensed_edges.columns) == [
        "source",
        "target",
        "boundary_length",
        "count",
        "edge_length",
    ]


def test_the_pipeline_adds_no_nondeterminism_of_its_own(mesh):
    """Everything the wrapper does itself is reproducible at a fixed seed.

    Simplification is switched off here, and it is the one step that is not:
    `fast_simplification.simplify` returns a different collapse list on every
    call, so `simplify_to_density` lands on a slightly different mesh each
    time and the domain count moves with it. That is a property of the
    simplifier, not of this wrapper, and `condensed_hks_pipeline` carries it
    too. Thresholding, label expansion and the recomputed `domain_` block are
    what is under test.
    """
    kwargs = dict(
        PIPELINE_KWARGS, simplify_target_density=None, simplify_target_reduction=None
    )
    first = condensed_spectral_pipeline(mesh, **kwargs)
    second = condensed_spectral_pipeline(mesh, **kwargs)

    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(first.mapping, second.mapping)
    pd.testing.assert_frame_equal(first.condensed_features, second.condensed_features)
    pd.testing.assert_frame_equal(first.condensed_edges, second.condensed_edges)
