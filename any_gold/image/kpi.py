from pathlib import Path
from typing import Callable

from torchvision.tv_tensors import Image as TvImage, Mask as TvMask

from any_gold.utils.load import load_torchvision_image, load_torchvision_mask
from any_gold.utils.synapse import SynapseZipBase


class KPITask1PatchLevel(SynapseZipBase):
    _ENTITIES: dict[str, dict[str, str]] = {
        "train": {
            "entity": "syn60249790",
            "name": "train",
        },
        "val": {
            "entity": "syn60249847",
            "name": "validation",
        },
        "test": {
            "entity": "syn63688309",
            "name": "test",
        },
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
        if split not in self._ENTITIES:
            raise ValueError(
                f"Split {split} is not available. Available splits are {self._ENTITIES.keys()}."
            )
        self.split = split
        self.entity = self._ENTITIES[split]["entity"]

        super().__init__(
            root=root,
            entity=self.entity,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            override=override,
        )

    def _setup(self) -> None:
        root = self.root / self._ENTITIES[self.split]["name"]
        if self.override or not root.exists():
            self.download()

        self.samples: list[tuple[Path, str]] = [
            (image_path, class_dir.name)
            for class_dir in (root / self._ENTITIES[self.split]["name"]).iterdir()
            for patch_dir in class_dir.iterdir()
            for image_path in (patch_dir / "img").glob("*.jpg")
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def get_image_path(self, index: int) -> Path:
        """Get the path of an image."""
        return self.samples[index][0]

    def __getitem__(self, index: int) -> tuple[TvImage, TvMask, str]:
        """Get an image and its corresponding mask."""
        image_path, target = self.samples[index]
        mask_path = image_path.parent.parent / f"mask/{image_path.stem[:-3]}mask.jpg"

        image = load_torchvision_image(image_path)
        mask = load_torchvision_mask(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask, target
