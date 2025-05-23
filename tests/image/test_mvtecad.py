from pathlib import Path

import pytest
import torch
from huggingface_hub import HfApi

from any_gold.image.mvtec_ad import MVTecADDataset
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

        image, mask, defect, target, index = dataset[0]
        assert index == 0, "Index is not as expected"
        assert image.shape == (1, 1, 1024, 1024), "Image shape is not as expected"
        assert mask.shape == (1, 1, 1024, 1024), "Mask shape is not as expected"
        assert target == torch.tensor(0), "Target is not as expected"
