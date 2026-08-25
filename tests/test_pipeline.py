import numpy as np
import pandas as pd
import pytest

from meshmash import simplify_to_density, surface_area, vertex_density
from meshmash.pipeline import condensed_hks_pipeline


def test_condensed_hks_pipeline_result_type(pipeline_result):
    assert pipeline_result is not None


def test_condensed_hks_pipeline_feature_columns(pipeline_result):
    assert isinstance(pipeline_result.condensed_features, pd.DataFrame)
    assert pipeline_result.condensed_features.shape[1] >= 1


def test_condensed_hks_pipeline_labels_length(pipeline_result):
    n_simple = pipeline_result.simple_mesh[0].shape[0]
    assert len(pipeline_result.simple_labels) == n_simple


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
