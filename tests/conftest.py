import pytest
from physgauss.data import SceneDataset, make_synthetic


@pytest.fixture
def scene_path(tmp_path):
    return make_synthetic(tmp_path / "scene", count=6, size=12, particles=5)


@pytest.fixture
def dataset(scene_path):
    return SceneDataset(scene_path)
