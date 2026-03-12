import pandas as pd


def test_condensed_hks_pipeline_result_type(pipeline_result):
    assert pipeline_result is not None


def test_condensed_hks_pipeline_feature_columns(pipeline_result):
    assert isinstance(pipeline_result.condensed_features, pd.DataFrame)
    assert pipeline_result.condensed_features.shape[1] >= 1


def test_condensed_hks_pipeline_labels_length(pipeline_result):
    n_simple = pipeline_result.simple_mesh[0].shape[0]
    assert len(pipeline_result.simple_labels) == n_simple
