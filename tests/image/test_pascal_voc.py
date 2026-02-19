import pytest

from pathlib import Path

from kagglehub import registry
from kagglehub.handle import DatasetHandle

from torch.utils.data import RandomSampler, DataLoader
from torchvision.transforms.v2 import Compose, Resize

from any_gold.image.pascal_voc import (
    PascalVOC2012Segmentation,
)
from any_gold.tools.image.utils import gold_multiclass_class_segmentation_collate_fn
from tests.conftest import TEST_DATASET_LOADING


class TestPascalVOC2012Segmentation:
    def test_handle(self):
        owner, dataset, _, version = PascalVOC2012Segmentation._HANDLE.split("/")
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
        dataset = PascalVOC2012Segmentation(
            root=Path("/storage/ml/pascal_voc"),
            split="train",
            transforms=Compose([Resize((224, 224))]),
        )

        assert len(dataset) == 1464, "Dataset length is not as expected"

        output = dataset[0]
        assert output["index"] == 0, "Index is not as expected"
        assert output["image"].shape == (
            3,
            224,
            224,
        ), "Image shape is not as expected"
        assert output["mask"].shape == (3, 224, 224), "Mask shape is not as expected"
        assert isinstance(output["labels"], set)
        assert len(output["labels"]) > 0

        sampler = RandomSampler(dataset, replacement=False, num_samples=5)
        dataloader = DataLoader(
            dataset,
            batch_size=5,
            sampler=sampler,
            num_workers=0,
            collate_fn=gold_multiclass_class_segmentation_collate_fn,
        )
        for batch in dataloader:
            assert batch["image"].shape == (5, 3, 224, 224)
