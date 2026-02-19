import shutil
from pathlib import Path
from typing import Callable

from PIL import Image
import torch
from torchvision.tv_tensors import Image as TvImage, Mask as TvMask

from any_gold.tools.image.utils import get_unique_pixel_values
from any_gold.utils.dataset import (
    MultiClassVisionSegmentationDataset,
    MultiClassVisionSegmentationOutput,
)
from any_gold.utils.kaggle import KaggleDataset


class PascalVOC2012SegmentationOutput(MultiClassVisionSegmentationOutput):
    """Output class for Pascal VOC Segmentation dataset from 2012."""

    pass


class PascalVOC2012Segmentation(MultiClassVisionSegmentationDataset, KaggleDataset):
    """Pascal VOC 2012 segmentation dataset from kaggle.

    The Pascal VOC 2012 segmentation dataset is introduced in
    [Pascal VOC challenge page](http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html)

    The dataset is a collection of 21 different classes of objects (20 object classes
    and 1 background class) with pixel-wise segmentation masks.

    The dataset is downloaded from [Kaggle](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset)
    and its data will be downloaded and stored in the specified root directory.

    There are 2 different splits available: train and val.

    Attributes:
        root: The root directory where the dataset is stored.
        split: The split of the dataset to use. Can be 'train' or 'val'. Default is 'train'.
        handle: The name of the dataset on Kaggle (same as _HANDLE).
        transform: A transform to apply to the images.
        target_transform: A transform to apply to the masks.
        transforms: A transform to apply to both images and masks.
            It cannot be set together with transform and target_transform.
        override: If True, will override the existing dataset in the root directory. Default is False.
        samples: A list of file paths to the images in the specified split.
    """

    _HANDLE = "gopalbhattrai/pascal-voc-2012-dataset/versions/1"
    _SPLITS = ("train", "val")
    _LABEL_MAPPING: dict[tuple[int, int, int], str] = {
        (0, 0, 0): "background",
        (128, 0, 0): "aeroplane",
        (0, 128, 0): "bicycle",
        (128, 128, 0): "bird",
        (0, 0, 128): "boat",
        (128, 0, 128): "bottle",
        (0, 128, 128): "bus",
        (128, 128, 128): "car",
        (64, 0, 0): "cat",
        (192, 0, 0): "chair",
        (64, 128, 0): "cow",
        (192, 128, 0): "diningtable",
        (64, 0, 128): "dog",
        (192, 0, 128): "horse",
        (64, 128, 128): "motorbike",
        (192, 128, 128): "person",
        (0, 64, 0): "pottedplant",
        (128, 64, 0): "sheep",
        (0, 192, 0): "sofa",
        (128, 192, 0): "train",
        (0, 64, 128): "tvmonitor",
        (224, 224, 192): "void",
    }

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
            raise ValueError(f"Split must be one of {self._SPLITS}, but got {split}.")

        self.split = split

        MultiClassVisionSegmentationDataset.__init__(
            self,
            root=root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        KaggleDataset.__init__(
            self,
            root=root,
            handle=self._HANDLE,
            override=override,
        )

    def _move_data_to_root(self, kaggle_cache: Path) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

        target = self.root / self.split
        source = kaggle_cache / "VOC2012_train_val/VOC2012_train_val"

        if target.exists():
            shutil.rmtree(target)

        target.mkdir(parents=True, exist_ok=True)

        if not source.exists():
            raise FileNotFoundError(
                f"Source directory {source} does not exist. "
                f"Please use override=True to download the dataset again."
            )

        # get the list of available images
        image_set_file = source / "ImageSets/Segmentation" / f"{self.split}.txt"
        if not image_set_file.exists():
            raise FileNotFoundError(
                f"Image set file {image_set_file} does not exist. "
                f"Please use override=True to download the dataset again."
            )
        image_filenames = set(image_set_file.read_text().splitlines())

        # move the different images and masks in the specified folder
        image_dir = source / "JPEGImages"
        mask_dir = source / "SegmentationClass"
        for image_filename in image_filenames:
            image_name = f"{image_filename}.jpg"
            mask_name = f"{image_filename}.png"

            image_path = image_dir / image_name
            mask_path = mask_dir / mask_name

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image file {image_path} does not exist. "
                    f"Please use override=True to download the dataset again."
                )
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask file {mask_path} does not exist. "
                    f"Please use override=True to download the dataset again."
                )

            shutil.copy(image_path, target)
            shutil.copy(mask_path, target)

    def _setup(self) -> None:
        root = self.root / self.split
        if self.override or not root.exists():
            self.download()

        self.samples = [image_path for image_path in root.glob("*.jpg")]

    def __len__(self) -> int:
        return len(self.samples)

    def get_raw(self, index: int) -> PascalVOC2012SegmentationOutput:
        """Get an image and its corresponding mask together with the index and labels."""
        image_path = self.samples[index]
        mask_path = image_path.parent / f"{image_path.stem}.png"

        image = TvImage(Image.open(image_path).convert("RGB"), dtype=torch.uint8)
        mask = TvMask(Image.open(mask_path).convert("RGB"), dtype=torch.uint8)

        unique_pixel_values = get_unique_pixel_values(mask)
        labels = set()
        for pixel_value in unique_pixel_values:
            assert len(pixel_value) == 3
            labels.add(self._LABEL_MAPPING[pixel_value])

        return PascalVOC2012SegmentationOutput(
            image=image, mask=mask, index=index, labels=labels
        )
