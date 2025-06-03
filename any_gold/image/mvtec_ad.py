from pathlib import Path
from typing import Callable

import torch
from torchvision.tv_tensors import Image as TvImage, Mask as TvMask

from any_gold.utils.dataset import (
    AnyVisionSegmentationDataset,
    AnyVisionSegmentationOutput,
)
from any_gold.utils.hugging_face import HuggingFaceDataset


class MVTecADOutput(AnyVisionSegmentationOutput):
    """Output class for MVTec Anomaly Detection dataset.

    It extends the AnyVisionSegmentationOutput class to include
    target (torch tensor indicating if the label is an anomaly or not).

    The label is the type of defect (`good` in absence of defect).
    """

    target: torch.Tensor


class MVTecADDataset(AnyVisionSegmentationDataset, HuggingFaceDataset):
    """MVTec Anomaly Detection Dataset.

    The Mvtec Anomaly Detection dataset is introduced in
    [The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf)

    The dataset is a collection of images and their corresponding segmentation masks for various
    manufacturing objects and type of defects.

    The dataset is downloaded from [Hugging Face](https://huggingface.co/datasets/TheoM55/mvtec_all_objects_split)
    and its data will be downloaded and stored in the specified root directory.

    For each category, there are two splits available: train and test.

    Args:
        root: The root directory where the dataset is stored.
        category: The category of the dataset (e.g., 'bottle', 'cable', 'capsule').
        split: The dataset split to use ('train' or 'test').
        path: The path of the dataset on Hugging Face.
        hf_split: it will be category.split.
        transform: A transform to apply to the images.
        target_transform: A transform to apply to the masks.
        transforms: A transform to apply to both images and masks.
        It cannot be set together with transform and target_transform.
        override: If True, will override the existing dataset in the root directory. Default is False.
        samples: the hugging face dataset in torch format.
    """

    _SPLITS = ("train", "test")
    _HUGGINGFACE_NAME = "TheoM55/mvtec_all_objects_split"

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
        override: bool = False,
    ) -> None:
        self.category = category
        self.split = split
        if split not in self._SPLITS:
            raise ValueError(f"Split must be one of {self._SPLITS}, but got {split}.")

        HuggingFaceDataset.__init__(
            self,
            root=root,
            path=self._HUGGINGFACE_NAME,
            hf_split=f"{self.category}.{self.split}",
            override=override,
        )
        AnyVisionSegmentationDataset.__init__(
            self,
            root=root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            override=override,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def get_raw(self, index: int) -> MVTecADOutput:
        """
        Get the image and its corresponding mask together with the label,
        the anomaly detection target and the index.
        """
        sample = self.samples[index]

        image = TvImage(sample["image_path"])

        mask = TvMask(
            sample["mask_path"]
            if sample["mask_path"] is not None
            else torch.zeros((1, *image.shape[-2:]), dtype=torch.uint8)
        )

        return MVTecADOutput(
            image=image,
            mask=mask,
            label=sample["defect"],
            target=sample["label"],
            index=index,
        )
