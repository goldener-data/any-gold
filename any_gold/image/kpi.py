from pathlib import Path
from typing import Callable


from any_gold.utils.dataset import (
    AnyVisionSegmentationDataset,
    AnyVisionSegmentationOutput,
)
from any_gold.utils.load import load_torchvision_image, load_torchvision_mask
from any_gold.utils.synapse import SynapseZipBase


class KPITask1PatchLevelOutput(AnyVisionSegmentationOutput):
    disease: str


class KPITask1PatchLevel(AnyVisionSegmentationDataset, SynapseZipBase):
    """KPI Task 1 Patch Level Dataset from Synapse.

    The KPI Task 1 Patch Level dataset is introduced in
    [KPIs 2024 Challenge: Advancing Glomerular Segmentation from Patch-to Slide-Level](https://arxiv.org/pdf/2502.07288)

    The dataset is a collection of images and their corresponding segmentation masks
    for glomeruli identification in various Chronic kidney diseases (CKD).

    The dataset is downloaded from [Synapse](https://www.synapse.org/Synapse:syn63688309) and its data will be downloaded
    and stored in the specified root directory.

    There are 3 different splits available: train, val, and test.

    Attributes:
        root: The root directory where the dataset is stored.
        split: The split of the dataset to use. Can be 'train', 'val', or 'test'. Default is 'train'.
        entity: The entity ID of the dataset on Synapse.
        transform: A transform to apply to the images.
        target_transform: A transform to apply to the masks.
        transforms: A transform to apply to both images and masks.
        It cannot be set together with transform and target_transform.
        override: If True, will override the existing dataset in the root directory. Default is False.
        samples: A list of tuples containing the image path and class name.
    """

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
        AnyVisionSegmentationDataset.__init__(
            self,
            root=root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            override=override,
        )

        if split not in self._ENTITIES:
            raise ValueError(
                f"Split {split} is not available. Available splits are {self._ENTITIES.keys()}."
            )
        self.split = split
        self.entity = self._ENTITIES[split]["entity"]
        SynapseZipBase.__init__(
            self,
            root=root,
            entity=self.entity,
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

    def get_raw(self, index: int) -> KPITask1PatchLevelOutput:
        image_path, disease = self.samples[index]
        mask_path = image_path.parent.parent / f"mask/{image_path.stem[:-3]}mask.jpg"

        image = load_torchvision_image(image_path)
        mask = load_torchvision_mask(mask_path)

        return KPITask1PatchLevelOutput(
            image=image, mask=mask, disease=disease, index=index
        )
