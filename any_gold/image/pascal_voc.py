import json
import shutil
from pathlib import Path
from typing import Callable

from PIL import Image
import torch
from torchvision.tv_tensors import Image as TvImage, Mask as TvMask

from any_gold.tools.image.utils import get_unique_pixel_values
from any_gold.utils.dataset import (
    AnyVisionSegmentationDataset,
    AnyVisionSegmentationOutput,
)
from any_gold.utils.kaggle import KaggleDataset


class PascalVOC2012SegmentationOutput(AnyVisionSegmentationOutput):
    """Output class for Pascal VOC Segmentation dataset from 2012.

    The label will always be the class in the image.
    """

    pass


class PascalVOC2012Segmentation(AnyVisionSegmentationDataset, KaggleDataset):
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
        _mask_mapping: A dictionary mapping the unique RGB values in the masks
        to their corresponding class labels.
    """

    _HANDLE = "gopalbhattrai/pascal-voc-2012-dataset/versions/1"
    _SPLITS = ("train", "val")
    _MASK_MAPPING_FILE = "mask_mapping.json"

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

        AnyVisionSegmentationDataset.__init__(
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

        self._setup_labels_values()

    def _setup_labels_values(self) -> None:
        mask_mapping_path = self.root / self._MASK_MAPPING_FILE
        if mask_mapping_path.exists():
            with open(mask_mapping_path, "r") as f:
                inverted_label_mapping = json.load(f)
                self._mask_mapping = {
                    tuple(value): int(label)
                    for label, value in inverted_label_mapping.items()
                }
        else:
            unique_values = set()
            for sample in range(len(self.samples)):
                image_path = self.samples[sample]
                mask_path = image_path.parent / f"{image_path.stem}.png"
                mask = TvMask(Image.open(mask_path).convert("RGB"), dtype=torch.uint8)
                unique_values.update(get_unique_pixel_values(mask))

            self._mask_mapping = {value: idx for idx, value in enumerate(unique_values)}

            with open(mask_mapping_path, "w") as file:
                json.dump(
                    {label: value for value, label in self._mask_mapping.items()},
                    file,
                    indent=2,
                )

    def __len__(self) -> int:
        return len(self.samples)

    def get_raw(self, index: int) -> PascalVOC2012SegmentationOutput:
        """Get an image and its corresponding mask together with the index and labels."""
        image_path = self.samples[index]
        mask_path = image_path.parent / f"{image_path.stem}.png"

        image = TvImage(Image.open(image_path).convert("RGB"), dtype=torch.uint8)
        mask = TvMask(Image.open(mask_path).convert("RGB"), dtype=torch.uint8)

        unique_pixel_values = get_unique_pixel_values(mask)
        mono_mask = torch.zeros_like(mask[0, :, :], dtype=torch.uint8)
        labels = set()
        for pixel_value in unique_pixel_values:
            new_value = self._mask_mapping[pixel_value]
            pixel_value_tensor = torch.tensor(pixel_value, dtype=torch.uint8).view(
                3, 1, 1
            )
            pixel_mask = torch.all(mask == pixel_value_tensor, dim=0)
            mono_mask[pixel_mask] = new_value
            labels.add(str(new_value))

        return PascalVOC2012SegmentationOutput(
            image=image, mask=TvMask(mono_mask.unsqueeze(0)), index=index, label=labels
        )
