import pytest

from torch.utils.data import RandomSampler, DataLoader
from any_gold import ISIC2018SegmentationDataset
from tests.conftest import TEST_DATASET_LOADING
from huggingface_hub import HfApi


class TestISIC2018SegmentationDataset:
    def test_huggingface_dataset_exists(self):
        api = HfApi()
        info = api.dataset_info(ISIC2018SegmentationDataset._HUGGINGFACE_NAME)
        assert info is not None

    @pytest.mark.skipif(
        not TEST_DATASET_LOADING, reason="TEST_DATASET_LOADING is not True"
    )
    def test_dataset(self):
        dataset = ISIC2018SegmentationDataset(
            root="/Users/boudoul/Desktop/dev/datasets/", split="val", override=True
        )
        assert len(dataset) == 100, "Dataset length is not as expected"
        sampler = RandomSampler(dataset, replacement=False, num_samples=5)
        dataloader = DataLoader(
            dataset,
            batch_size=5,
            sampler=sampler,
            num_workers=0,
        )
        for batch in dataloader:
            (
                batch["image"].shape == (5, 3, 1022, 767),
                "Batch image is not as expected",
            )
