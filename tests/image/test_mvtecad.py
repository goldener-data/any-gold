from pathlib import Path

import pytest

from huggingface_hub import HfApi

import torch
from torch.utils.data import RandomSampler, DataLoader

from any_gold import MVTecADDataset
from tests.conftest import TEST_DATASET_LOADING


class TestMVTecADDataset:
    def test_huggingface_dataset_exists(self):
        api = HfApi()
        info = api.dataset_info(MVTecADDataset._HUGGINGFACE_NAME)
        assert info is not None

    @pytest.mark.skipif(
        not TEST_DATASET_LOADING, reason="TEST_DATASET_LOADING is not True"
    )
    def test_dataset(self):
        dataset = MVTecADDataset(
            root=Path("/storage/ml/mvtecad"),
            category="zipper",
        )

        assert len(dataset) == 240, "Dataset length is not as expected"

        output = dataset[0]
        assert output["index"] == 0, "Index is not as expected"
        assert output["image"].shape == (
            1,
            1024,
            1024,
        ), "Image shape is not as expected"
        assert output["mask"].shape == (1, 1024, 1024), "Mask shape is not as expected"
        assert output["target"] == torch.tensor(0), "Target is not as expected"

        sampler = RandomSampler(dataset, replacement=False, num_samples=5)
        dataloader = DataLoader(
            dataset,
            batch_size=5,
            sampler=sampler,
            num_workers=0,
        )
        for batch in dataloader:
            (
                batch["image"].shape == (5, 1, 1024, 1024),
                "Batch image shape is not as expected",
            )
