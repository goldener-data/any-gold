from pathlib import Path
from typing import Callable

from PIL import Image
import pandas as pd
import torch
from torchvision.tv_tensors import Image as TvImage, Mask as TvMask

from any_gold.utils.dataset import (
    SingleClassVisionSegmentationDataset,
    SingleClassVisionSegmentationOutput,
)
from any_gold.utils.zenodo import ZenodoZipBase


class PlantSegOutput(SingleClassVisionSegmentationOutput):
    """Output class for PlantSeg dataset.

    It extends the SingleClassVisionSegmentationOutput class to include plant species.
    The label will be the disease on the plant.
    """

    plant: str


class PlantSeg(SingleClassVisionSegmentationDataset, ZenodoZipBase):
    """PlantSeg Dataset from Zenodo.

    The PlantSeg dataset is introduced in
    [PlantSeg: A Large-Scale In-the-wild Dataset for Plant Disease Segmentation](https://arxiv.org/abs/2409.04038)

    The dataset is a collection of images and their corresponding segmentation masks for plant diseases (1 mask per image).
    The dataset is available in three versions, each with different images and masks.

    The dataset is downloaded from [Zenodo](https://zenodo.org/records/14935094)
    and its data will be downloaded and stored in the specified root directory.

    There are 3 different splits available: train, val, and test.

    Attributes:
        root: The root directory where the dataset is stored.
        version: The version of the dataset to use. Default is 3.
        split: The split of the dataset to use. Can be 'train', 'val', or 'test'. Default is 'train'.
        record_id: The record ID of the dataset on Zenodo.
        name: The name of the dataset on Zenodo.
        transform: A transform to apply to the images.
        target_transform: A transform to apply to the masks.
        transforms: A transform to apply to both images and masks.
        It cannot be set together with transform and target_transform.
        override: If True, will override the existing dataset in the root directory. Default is False.
        samples: A list of tuples containing the image path, plant name, and disease name.
    """

    _VERSIONS = {
        1: {
            "record_id": "13762907",
            "name": "plantseg.zip",
        },
        2: {
            "record_id": "13958858",
            "name": "plantsegv2.zip",
        },
        3: {
            "record_id": "14935094",
            "name": "plantsegv3.zip",
            "metadata": "Metadatav2.csv",
        },
    }

    def __init__(
        self,
        root: str | Path,
        version: int = 3,
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
        override: bool = False,
    ) -> None:
        SingleClassVisionSegmentationDataset.__init__(
            self,
            root=root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

        if version not in self._VERSIONS:
            raise ValueError(
                f"Version {version} is not available. Available versions are {list(self._VERSIONS.keys())}."
            )
        self.version = version

        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Split {split} is not available. Available splits are ['train', 'val', 'test']."
            )
        self.split = split

        ZenodoZipBase.__init__(
            self,
            root=root,
            record_id=self._VERSIONS[version]["record_id"],
            name=self._VERSIONS[version]["name"],
            override=override,
        )

        self.samples: list[tuple[Path, str, str]]

    def _setup(self) -> None:
        if self.override or not self.root.exists():
            self.download()

        self.samples = []
        metadata = pd.read_csv(
            self.root
            / f"plantsegv{self.version}/{self._VERSIONS[self.version]['metadata']}"
        )
        for image_path in sorted(
            (self.root / f"plantsegv{self.version}/images/{self.split}").glob("*.jpg")
        ):
            row = metadata[metadata["Name"] == image_path.name]
            if len(row) != 1:
                raise ValueError(
                    "Expected exactly one row per image in the PlantSeg metadata file."
                )

            self.samples.append(
                (image_path, row.iloc[0]["Plant"], row.iloc[0]["Disease"])
            )

    def __len__(self) -> int:
        return len(self.samples)

    def get_raw(self, index: int) -> PlantSegOutput:
        """Get the image and its corresponding mask together with the plant species, disease and index."""
        image_path, plant, disease = self.samples[index]
        mask_path = (
            image_path.parent.parent.parent
            / f"annotations/{self.split}/{image_path.stem}.png"
        )

        image = TvImage(Image.open(image_path).convert("RGB"), dtype=torch.uint8)
        mask = (
            TvMask(Image.open(mask_path).convert("L"), dtype=torch.uint8)
            if mask_path is not None
            else torch.zeros((1, *image.shape[-2:]), dtype=torch.uint8)
        )

        return PlantSegOutput(
            index=index,
            image=image,
            mask=mask,
            plant=plant,
            label=disease,
        )
