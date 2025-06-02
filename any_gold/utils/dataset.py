from abc import abstractmethod
from pathlib import Path
from typing import Callable, OrderedDict, Any

from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.tv_tensors import Image as TvImage, Mask as TvMask


class AnyOutput(OrderedDict):
    def __init__(self, *, index: int, **kwargs: Any):
        super().__init__(index=index, **kwargs)


class AnyDataset(Dataset):
    """Base class for any dataset.

    Attributes:
        root: The root directory where the dataset is stored.
        override: If True, will override the existing dataset in the root directory.
    """

    def __init__(
        self,
        root: str | Path,
        override: bool = False,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.override = override

    @abstractmethod
    def get_raw(self, index: int) -> AnyOutput:
        """Get the raw data for the given index."""

    @abstractmethod
    def __get_item__(self, index: int) -> AnyOutput:
        """Get the transformed data for the given index."""

    @abstractmethod
    def __len__(self) -> int:
        """Get the length of the dataset."""


class AnyVisionSegmentationOutput(AnyOutput):
    def __init__(self, *, index: int, image: TvImage, mask: TvMask, **kwargs: Any):
        super().__init__(index=index, image=image, mask=mask, **kwargs)


class AnyVisionSegmentationDataset(VisionDataset, AnyDataset):
    """Base class for any vision dataset.

    Attributes:
        root: The root directory where the dataset is stored.
        transform: A transform to apply to the images.
        target_transform: A transform to apply to the masks.
        transforms: A transform to apply to both images and masks.
        It cannot be set together with transform and target_transform.
        override: If True, will override the existing dataset in the root directory. Default is False.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
        override: bool = False,
    ) -> None:
        AnyDataset.__init__(self, root, override)

        VisionDataset.__init__(
            self,
            root=root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

    @abstractmethod
    def get_raw(self, index: int) -> AnyVisionSegmentationOutput:
        """Get the raw data for the given index."""

    def __getitem__(self, index: int) -> AnyVisionSegmentationOutput:
        """Get the transformed image and its corresponding information."""
        output = self.get_raw(index)

        image = output["image"]
        mask = output["mask"]

        if self.transform:
            output["image"] = self.transform(image)
        if self.target_transform:
            output["mask"] = self.target_transform(mask)
        if self.transforms:
            output["image"], output["mask"] = self.transforms(image, mask)

        return output


class AnyRawDataset:
    """Wrapper allowing to load raw data from a dataset.

    Attributes:
        dataset: The dataset to wrap.
    """

    def __init__(
        self,
        dataset: AnyDataset,
    ) -> None:
        self._dataset = dataset

    def __iter__(self):
        return iter(self._dataset)

    def __get_item__(self, index: int) -> AnyOutput:
        """Get the raw data of the dataset for the given index."""
        return self._dataset.get_raw(index)

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self._dataset)
