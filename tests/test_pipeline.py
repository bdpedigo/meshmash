import numpy as np
import pandas as pd
import pytest

from meshmash import simplify_to_density, surface_area, vertex_density
from meshmash.pipelines import condensed_hks_pipeline


def test_condensed_hks_pipeline_result_type(pipeline_result):
    assert pipeline_result is not None


def test_condensed_hks_pipeline_feature_columns(pipeline_result):
    assert isinstance(pipeline_result.condensed_features, pd.DataFrame)
    assert pipeline_result.condensed_features.shape[1] >= 1


def test_condensed_hks_pipeline_labels_length(pipeline_result):
    n_simple = pipeline_result.simple_mesh[0].shape[0]
    assert len(pipeline_result.simple_labels) == n_simple


def test_condensed_hks_pipeline_labels_are_int32(pipeline_result):
    assert pipeline_result.simple_labels.dtype == np.int32
    assert pipeline_result.labels.dtype == np.int32


def test_pipeline_rejects_both_simplify_targets(mesh):
    with pytest.raises(ValueError):
        condensed_hks_pipeline(
            mesh,
            simplify_target_reduction=0.7,
            simplify_target_density=1e-5,
            n_jobs=1,
        )


def test_simplify_to_density_hits_target(mesh):
    target = vertex_density(mesh) / 4
    vertices, faces, mapping = simplify_to_density(mesh, target)
    assert len(mapping) == len(mesh[0])
    assert mapping.max() < len(vertices)
    assert vertex_density((vertices, faces)) <= target * 1.05


def test_the_hks_pipeline_simplifies_reproducibly_but_featurizes_otherwise(mesh):
    """What `condensed_hks_pipeline` actually offers, pinned both ways.

    Its simplification is deterministic, under the
    `fast-simplification>=0.1.9` floor pyproject.toml requires. Its features
    are not, because the pipeline exposes no `seed`: ARPACK draws its own
    starting vector, the spectral bisection draws its own cut, and Ward flips
    merges on the resulting ties. Use `condensed_spectral_pipeline` for a
    reproducible composite run. See TASK-15.
    """
    kwargs = dict(n_components=4, max_eigenvalue=1e-8, simplify_target_reduction=0.7)
    first = condensed_hks_pipeline(mesh, n_jobs=1, **kwargs)
    second = condensed_hks_pipeline(mesh, n_jobs=1, **kwargs)

    np.testing.assert_array_equal(first.simple_mesh[0], second.simple_mesh[0])
    np.testing.assert_array_equal(first.simple_mesh[1], second.simple_mesh[1])
    np.testing.assert_array_equal(first.mapping, second.mapping)


def test_degenerate_mesh_is_robust():
    # Three collinear points -> a single zero-area face.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    assert surface_area((vertices, faces)) == 0.0
    assert vertex_density((vertices, faces)) == float("inf")
    assert (
        vertex_density((np.empty((0, 3), np.float32), np.empty((0, 3), np.int32)))
        == 0.0
    )

    v, f, mapping = simplify_to_density((vertices, faces), target_density=1e-3)
    assert len(v) == len(vertices)
    assert np.array_equal(mapping, np.arange(len(vertices)))
