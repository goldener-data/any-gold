import pytest

from kagglehub import registry
from pathlib import Path

from kagglehub.handle import DatasetHandle

from any_gold.image.deepglobe import DeepGlobeRoadExtraction
from tests.conftest import TEST_DATASET_LOADING


class TestDeepGlobeRoadExtraction:
    def test_handle(self):
        owner, dataset, _, version = DeepGlobeRoadExtraction._HANDLE.split("/")
        registry.dataset_resolver(
            DatasetHandle(
                owner=owner,
                dataset=dataset,
                version=int(version),
            )
        )

    @pytest.mark.skipif(
        not TEST_DATASET_LOADING, reason="TEST_DATASET_LOADING is not True"
    )
    def test_dataset(self):
        dataset = DeepGlobeRoadExtraction(
            root=Path("/storage/ml/deepglobe_road_extraction"),
            split="train",
        )

        assert len(dataset) == 6226, "Dataset length is not as expected"

        image, mask, index = dataset[0]
        assert index == 0, "Index is not as expected"
        assert image.shape == (1, 3, 1024, 1024), "Image shape is not as expected"
        assert mask.shape == (1, 1, 1024, 1024), "Mask shape is not as expected"
        assert dataset.get_image_path(0) == Path(
            "/storage/ml/deepglobe_road_extraction/train/839673_sat.jpg"
        ), "Image path is not as expected"
