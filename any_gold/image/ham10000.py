from pathlib import Path
from typing import Callable
import shutil
import pandas as pd
from PIL import Image
import torch
from torchvision.tv_tensors import Image as TvImage

from any_gold.utils.dataset import AnyOutput
from any_gold.utils.kaggle import KaggleDataset


class HAM10000Output(AnyOutput):
    """
    Output class for Human Against Machine Skin Lesion dataset.
    """

    image: TvImage
    label: str
    class_index: int


class HAM10000Dataset(KaggleDataset):
    """HAM10000 skin lesion dataset from Kaggle.

    The HAM10000 dataset ("Human Against Machine with 10000 training images") is a collection of dermatoscopic images
    used for skin lesion classification. It includes images of common pigmented skin lesions and was released as part
    of the ISIC 2018 challenge.

    The dataset is downloaded from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
    and stored in the specified root directory.

    To avoid data leakage, the dataset is split based on unique lesion IDs so that all images of a given lesion
    belong to a single split. The available splits are 'train', 'val', and 'test'.

    Attributes:
        root: The root directory where the dataset is stored.
        split: The split of the dataset to use. Can be 'train', 'val', or 'test'. Default is 'train'.
        handle: The name of the dataset on Kaggle (same as _HANDLE).
        transform: A transform to apply to the images.
        override: If True, will override the existing dataset in the root directory. Default is False.
        samples: A list of image identifiers (image_id) corresponding to the selected split.
        labels: The class label (e.g., 'mel', 'nv', 'bkl', etc.) for each image.
    """

    _HANDLE = "kmader/skin-cancer-mnist-ham10000/dataset/1"
    _SPLITS = {"part1": "HAM10000_images_part_1", "part2": "HAM10000_images_part_2"}

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable | None = None,
        override: bool = False,
    ) -> None:
        self.transform = transform
        self.split = split

        KaggleDataset.__init__(self, root=root, handle=self._HANDLE, override=override)

    def _move_data_to_root(self, kaggle_cache: Path) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

        target = self.root / self._SPLITS[self.split]
        source = kaggle_cache / self._SPLITS[self.split]

        if target.exists():
            shutil.rmtree(target)

        if not source.exists():
            raise FileNotFoundError(
                f"Source directory {source} does not exist. "
                f"Please use override=True to download the dataset again."
            )

        shutil.move(source, target)

    def _setup(self) -> None:
        metadata_path = self.root / "HAM10000_metadata.csv"

        if self.override or not metadata_path.exists():
            self.download()

        self.metadata = pd.read_csv(metadata_path)

        self.samples = self.metadata["image_id"].tolist()
        self.labels = self.metadata["dx"].tolist()
        self.image_dir = self.root / self._SPLITS[self.split]
        self.class_names = sorted(set(self.labels))
        self.label_to_index = {label: idx for idx, label in enumerate(self.class_names)}

        valid_samples = []
        valid_labels = []
        for sample, label in zip(self.samples, self.labels):
            img_path = self.image_dir / f"{sample}.jpg"
            if img_path.exists():
                valid_samples.append(sample)
                valid_labels.append(label)

        self.samples = valid_samples
        self.labels = valid_labels

    def __len__(self) -> int:
        return len(self.samples)

    def get_image_path(self, index: int) -> Path:
        return self.image_dir / f"{self.samples[index]}.jpg"

    def get_raw(self, index: int) -> HAM10000Output:
        img_path = self.get_image_path(index)
        label = self.labels[index]
        class_index = self.label_to_index[label]

        image = TvImage(Image.open(img_path).convert("RGB"), dtype=torch.uint8)

        return HAM10000Output(
            index=index, image=image, label=label, class_index=class_index
        )

    def __getitem__(self, index: int) -> HAM10000Output:
        output = self.get_raw(index)
        if self.transform:
            output["image"] = self.transform(output["image"])
        return output
