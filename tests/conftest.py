import pytest

from meshmash import fetch_sample_mesh
from meshmash.pipeline import condensed_hks_pipeline

_FAST_PIPELINE_KWARGS = dict(
    n_components=4,
    max_eigenvalue=1e-8,
    simplify_target_reduction=0.7,
    n_jobs=1,
)


@pytest.fixture(scope="session")
def mesh():
    return fetch_sample_mesh("microns_dendrite_sample")


@pytest.fixture(scope="session")
def pipeline_result(mesh):
    return condensed_hks_pipeline(mesh, **_FAST_PIPELINE_KWARGS)
