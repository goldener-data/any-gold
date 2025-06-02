import pytest

from pathlib import Path

from kagglehub import registry
from kagglehub.handle import DatasetHandle

from torch.utils.data import RandomSampler, DataLoader

from any_gold import DeepGlobeRoadExtraction
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

        output = dataset[0]
        assert output["index"] == 0, "Index is not as expected"
        assert output["image"].shape == (
            3,
            1024,
            1024,
        ), "Image shape is not as expected"
        assert output["mask"].shape == (1, 1024, 1024), "Mask shape is not as expected"
        assert dataset.get_image_path(0) == Path(
            "/storage/ml/deepglobe_road_extraction/train/839673_sat.jpg"
        ), "Image path is not as expected"

        sampler = RandomSampler(dataset, replacement=False, num_samples=5)
        dataloader = DataLoader(
            dataset,
            batch_size=5,
            sampler=sampler,
            num_workers=0,
        )
        for batch in dataloader:
            (
                batch["image"].shape == (5, 3, 1024, 1024),
                "Batch image shape is not as expected",
            )
