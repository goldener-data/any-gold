from pathlib import Path
from typing import Callable

import torch
from torchvision.transforms.v2 import PILToTensor
from torchvision.tv_tensors import Image as TvImage, Mask as TvMask
from PIL import Image as PILImage

from any_gold.utils.zenodo import ZenodoZipBase


PLANTSEG_VERSIONS = {
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
    },
}


class PlantSeg(ZenodoZipBase):
    """
    PlantSeg Dataset from Zenodo.
    """

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
        if version not in PLANTSEG_VERSIONS:
            raise ValueError(
                f"Version {version} is not available. Available versions are {list(PLANTSEG_VERSIONS.keys())}."
            )
        self.version = version

        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Split {split} is not available. Available splits are ['train', 'val', 'test']."
            )
        self.split = split

        self.record_id = PLANTSEG_VERSIONS[version]["record_id"]
        self.name = PLANTSEG_VERSIONS[version]["name"]

        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            override=override,
        )

        data_folder = self.root / f"plantsegv{self.version}"
        self.image_files = list((data_folder / f"images/{self.split}").glob("*.jpg"))
        self.mask_folder = data_folder / f"annotations/{self.split}"

    def __len__(self) -> int:
        """
        Return the number of images in the dataset.
        """
        return len(self.image_files)


    def _load_image(self, path: Path) -> TvImage:
        pil_image = PILImage.open(path).convert("RGB")
        return TvImage(pil_image).unsqueeze(0)

    def _load_mask(self, path: Path) -> TvMask:
        pil_mask = PILImage.open(path).convert("L")
        return TvMask(pil_mask).unsqueeze(0)

    def get_image_path(self, index: int) -> Path:
        """
        Get the path of an image.
        """
        return self.image_files[index]

    def __getitem__(self, index: int) -> tuple[TvImage, TvMask]:
        """
        Get an image and its corresponding mask.
        """
        image_path = self.image_files[index]
        mask_path = self.mask_folder / f"{image_path.stem}.png"

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask





