import pytest
from pathlib import Path

from torchvision.transforms import v2
from zenodo_client import Zenodo

from torch.utils.data import RandomSampler, DataLoader

from any_gold import PlantSeg
from tests.conftest import TEST_DATASET_LOADING, ZENODO_API_TOKEN


class TestPlantSeg:
    def test_records(self):
        zenodo = Zenodo(access_token=ZENODO_API_TOKEN)
        record_v1 = zenodo.get_latest_record(PlantSeg._VERSIONS[1]["record_id"])
        assert record_v1 == PlantSeg._VERSIONS[1]["record_id"]
        record_v2 = zenodo.get_latest_record(PlantSeg._VERSIONS[2]["record_id"])
        assert record_v2 == PlantSeg._VERSIONS[2]["record_id"]
        record_v3 = zenodo.get_latest_record(PlantSeg._VERSIONS[3]["record_id"])
        assert record_v3 == PlantSeg._VERSIONS[3]["record_id"]

    @pytest.mark.skipif(
        not TEST_DATASET_LOADING, reason="TEST_DATASET_LOADING is not True"
    )
    def test_dataset(self):
        dataset = PlantSeg(
            root=Path("/storage/ml/plantseg"),
            version=3,
            split="train",
            transforms=v2.Resize((853, 640)),
        )

        assert len(dataset) == 7916, "Dataset length is not as expected"

        output = dataset[0]
        assert output["index"] == 0, "Index is not as expected"
        assert output["image"].shape == (3, 853, 640), "Image shape is not as expected"
        assert output["mask"].shape == (1, 853, 640), "Mask shape is not as expected"
        assert output["plant"] == "Apple"
        assert output["disease"] == "apple mosaic virus"

        sampler = RandomSampler(dataset, replacement=False, num_samples=5)
        dataloader = DataLoader(
            dataset,
            batch_size=5,
            sampler=sampler,
            num_workers=0,
        )
        for batch in dataloader:
            (
                batch["image"].shape == (5, 3, 853, 640),
                "Batch image shape is not as expected",
            )
