import numpy as np
import pytest
import torch
from pathlib import Path
from torchvision.tv_tensors import Image as TvImage, Mask as TvMask
import fiftyone as fo
from any_gold.tools.image.fiftyone_conversion import (
    build_fo_detections_from_connected_components,
    build_fo_sample_from_any_vision_segmentation_output,
    build_fo_dataset_from_any_vision_segmentation_dataset,
)
from any_gold.tools.image.connected_component import ConnectedComponent
from typing import Callable

from any_gold.utils.dataset import (
    AnyVisionSegmentationOutput,
    AnyVisionSegmentationDataset,
)


@pytest.fixture
def mask() -> TvMask:
    return TvMask(torch.ones((1, 10, 10), dtype=torch.uint8))


@pytest.fixture
def image() -> TvImage:
    return TvImage(torch.ones((3, 10, 10), dtype=torch.uint8))


class TestBuildFoDetectionsFromConnectedComponents:
    def test_empty_components_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            build_fo_detections_from_connected_components([], "label", 0, tmp_path)

    def test_single_component(self, tmp_path: Path, mask: TvMask) -> None:
        cc = ConnectedComponent(
            mask=mask.permute(1, 2, 0).numpy(),
            bounding_box=np.array([0, 0, 10, 10], dtype=np.uint32),
            area=100,
        )
        detections = build_fo_detections_from_connected_components(
            [cc], "cat", 1, tmp_path
        )
        assert isinstance(detections, fo.Detections)
        assert len(detections.detections) == 1
        assert detections.detections[0].label == "cat"
        assert Path(detections.detections[0].mask_path).exists()


class TestBuildFoSampleFromAnyVisionSegmentationOutput:
    def test_sample_conversion(
        self, tmp_path: Path, image: TvImage, mask: TvMask
    ) -> None:
        sample = AnyVisionSegmentationOutput(
            label="dog",
            index=42,
            image=image,
            mask=mask,
        )
        fo_sample = build_fo_sample_from_any_vision_segmentation_output(
            sample, tmp_path
        )
        assert isinstance(fo_sample, fo.Sample)
        assert Path(fo_sample.filepath).exists()
        assert fo_sample.ground_truth is not None
        assert fo_sample.metadata.width == 10
        assert fo_sample.metadata.height == 10
        assert fo_sample.metadata.index == 42

    def test_sample_conversion_with_no_object(
        self,
        tmp_path: Path,
        image: TvImage,
    ) -> None:
        sample = AnyVisionSegmentationOutput(
            label="dog",
            index=42,
            image=image,
            mask=TvMask(torch.zeros((1, 10, 10), dtype=torch.uint8)),  # No mask
        )
        fo_sample = build_fo_sample_from_any_vision_segmentation_output(
            sample, tmp_path
        )
        assert isinstance(fo_sample, fo.Sample)
        assert Path(fo_sample.filepath).exists()
        assert fo_sample.ground_truth is None
        assert fo_sample.metadata.width == 10
        assert fo_sample.metadata.height == 10
        assert fo_sample.metadata.index == 42


@pytest.fixture
def dummy_dataset(
    tmp_path: Path, image: TvImage, mask: TvMask
) -> AnyVisionSegmentationDataset:
    class DummyDataset(AnyVisionSegmentationDataset):
        def __init__(
            self,
            root: str | Path,
            transform: Callable | None = None,
            target_transform: Callable | None = None,
            transforms: Callable | None = None,
            override: bool = False,
        ) -> None:
            super().__init__(
                root=root,
                transform=transform,
                target_transform=target_transform,
                transforms=transforms,
                override=override,
            )

        def __len__(self) -> int:
            return 2

        def get_raw(self, idx: int) -> AnyVisionSegmentationOutput:
            return AnyVisionSegmentationOutput(
                label="a",
                image=image,
                mask=mask
                if idx == 0
                else TvMask(torch.zeros((1, 10, 10), dtype=torch.uint8)),
                index=idx,
            )

    return DummyDataset(root=tmp_path)


class TestBuildFoDatasetFromAnyVisionSegmentationDataset:
    def test_dataset_conversion(
        self, tmp_path: Path, dummy_dataset: AnyVisionSegmentationDataset, mask: TvMask
    ) -> None:
        fo_dataset = build_fo_dataset_from_any_vision_segmentation_dataset(
            dummy_dataset,
            dataset_name="test_ds",
            save_dir=tmp_path,
            batch_size=1,
            num_workers=0,
            seed=123,
            persistent=False,
            overwrite=True,
        )
        assert isinstance(fo_dataset, fo.Dataset)
        assert len(fo_dataset) == 2

    def test_dataset_conversion_with_sampler(
        self, tmp_path: Path, dummy_dataset: AnyVisionSegmentationDataset, mask: TvMask
    ) -> None:
        fo_dataset = build_fo_dataset_from_any_vision_segmentation_dataset(
            dummy_dataset,
            dataset_name="test_ds",
            save_dir=tmp_path,
            num_samples=1,  # Use all samples
            batch_size=1,
            num_workers=0,
            seed=123,
            persistent=False,
            overwrite=True,
        )
        assert isinstance(fo_dataset, fo.Dataset)
        assert len(fo_dataset) == 1
