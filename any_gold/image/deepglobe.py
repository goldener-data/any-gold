import shutil
from pathlib import Path
from typing import Callable

import PIL
import torch
from torchvision.tv_tensors import Image as TvImage, Mask as TvMask

from any_gold.utils.dataset import (
    AnyVisionSegmentationDataset,
    AnyVisionSegmentationOutput,
)
from any_gold.utils.kaggle import KaggleDataset


class DeepGlobeRoadExtractionOutput(AnyVisionSegmentationOutput):
    """Output class for DeepGlobe Road Extraction dataset.

    The label will always be `road`.
    """

    pass


class DeepGlobeRoadExtraction(AnyVisionSegmentationDataset, KaggleDataset):
    """Deepglobe road extraction dataset from kaggle.

    The DeepGlobe road extraction dataset is introduced in
    [DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images](https://arxiv.org/pdf/1805.06561)

    The dataset is a collection of satellite images with road to be extracted from.
    Only the training set is integrating road location (1 mask per image)

    The dataset is downloaded from [Kaggle](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
    and its data will be downloaded and stored in the specified root directory.

    There are 3 different splits available: train, val, and test.

    Attributes:
        root: The root directory where the dataset is stored.
        split: The split of the dataset to use. Can be 'train', 'val', or 'test'. Default is 'train'.
        handle: The name of the dataset on Kaggle (same as _HANDLE).
        transform: A transform to apply to the images.
        target_transform: A transform to apply to the masks.
        transforms: A transform to apply to both images and masks.
        It cannot be set together with transform and target_transform.
        override: If True, will override the existing dataset in the root directory. Default is False.
        samples: A list of file paths to the satellite images in the specified split.
    """

    _HANDLE = "balraj98/deepglobe-road-extraction-dataset/versions/2"

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
        override: bool = False,
    ) -> None:
        AnyVisionSegmentationDataset.__init__(
            self,
            root=root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Split {split} is not available. Available splits are ['train', 'val', 'test']."
            )
        self.split = split

        KaggleDataset.__init__(
            self,
            root=root,
            handle=self._HANDLE,
            override=override,
        )

    def _move_data_to_root(self, dataset_directory: Path) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for file in dataset_directory.rglob("*"):
            if file.is_file():
                relative_path = file.relative_to(dataset_directory)
                target_path = self.root / relative_path
                if target_path.parent.stem == "valid":
                    target_path = target_path.parent.parent / f"val/{target_path.name}"
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file), str(target_path))

    def _setup(self) -> None:
        if self.override or not self.root.exists():
            self.download()

        self.samples = [
            image_path for image_path in (self.root / self.split).glob("*_sat.jpg")
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def get_image_path(self, index: int) -> Path:
        """Get the path of an image."""
        return self.samples[index]

    def get_raw(self, index: int) -> DeepGlobeRoadExtractionOutput:
        """Get an image and its corresponding mask together with the index.

        If the split is not 'train', the mask will be None.
        """
        image_path = self.samples[index]
        mask_path = image_path.parent / f"{image_path.stem[:-3]}mask.png"

        image = TvImage(PIL.Image.open(image_path).convert("RGB"), dtype=torch.uint8)
        mask = (
            TvMask(PIL.Image.open(mask_path).convert("L"), dtype=torch.uint8)
            if mask_path is not None
            else torch.zeros((1, *image.shape[-2:]), dtype=torch.uint8)
        )

        return DeepGlobeRoadExtractionOutput(
            image=image, mask=mask, index=index, label="road"
        )
