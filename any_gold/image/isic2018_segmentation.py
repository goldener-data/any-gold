from pathlib import Path
from typing import Callable
import torch
from torchvision.tv_tensors import Image as TvImage, Mask as TvMask

from any_gold.utils.dataset import (
    AnyVisionSegmentationDataset,
    AnyVisionSegmentationOutput,
)
from any_gold.utils.hugging_face import HuggingFaceDataset


class ISIC2018SegmentationOutput(AnyVisionSegmentationOutput):
    """
    Output class for ISIC2018 Skin Lesion Segmentation dataset.
    """

    pass


class ISIC2018SegmentationDataset(AnyVisionSegmentationDataset, HuggingFaceDataset):
    """Skin Lesion Segmentation Dataset.

    This dataset is part of the ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection challenge,
    specifically for the lesion boundary segmentation task. It contains dermoscopic images and
    corresponding expert-annotated binary masks, enabling training and evaluation of skin lesion
    segmentation algorithms.

    The dataset is downloaded from [Hugging Face](https://huggingface.co/datasets/surajbijjahalli/ISIC2018)
    and stored in the specified root directory.
    There are 3 different splits available: 'train', 'val', and 'test'.

    Reference:
    Noel Codella et al., "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted
    by the International Skin Imaging Collaboration (ISIC)", 2018, arXiv:1902.03368.
    https://doi.org/10.48550/arXiv.1902.03368

    Attributes:
    root: The root directory where the dataset is stored.
    split: The split of the dataset to use. Can be 'train', 'val', or 'test'. Default is 'train'.
    handle: The name of the dataset on Hugging Face (same as _HUGGINGFACE_NAME).
    transform: A transform to apply to the images.
    target_transform: A transform to apply to the masks.
    transforms: A transform to apply to both images and masks.
    override: If True, will override the existing dataset in the root directory. Default is False.
    samples: A list of image identifiers corresponding to the selected split.
    """

    _HUGGINGFACE_NAME = "surajbijjahalli/ISIC2018"

    _SPLITS = {"train": "train", "val": "validation", "test": "test"}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
        override: bool = False,
    ) -> None:
        if split not in self._SPLITS:
            raise ValueError(f"split must be one of {self._SPLITS}, but got {split}")

        self.split = split

        HuggingFaceDataset.__init__(
            self,
            path=self._HUGGINGFACE_NAME,
            hf_split=f"{self._SPLITS[self.split]}",
            override=override,
        )

        AnyVisionSegmentationDataset.__init__(
            self,
            root=root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def get_raw(self, index: int) -> ISIC2018SegmentationOutput:
        """
        Get an image and its corresponding mask together
        """
        sample = self.samples[index]
        image = TvImage(sample["image"])
        mask = (
            TvMask(sample["label"])
            if sample["label"] is not None
            else torch.zeros((1, *image.shape[-2:]), dtype=torch.uint8)
        )
        return ISIC2018SegmentationOutput(
            image=image, mask=mask, index=index, label="lesion"
        )
