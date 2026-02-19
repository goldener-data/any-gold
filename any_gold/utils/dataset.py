from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Callable, TypedDict, Any

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset
from torchvision.tv_tensors import Image as TvImage, Mask as TvMask
from tqdm import tqdm

from any_gold.tools.image.connected_component import (
    extract_connected_components_from_binary_mask,
)
from any_gold.tools.image.stats import vision_segmentation_stats


class AnyOutput(TypedDict):
    """Base class for any output from a dataset.

    It specifies the output format for any dataset. All any gold dataset is expected to return an instance of this class.
    At least the index of the data is required, but it can also contain
    other information such as image, annotation, etc...
    """

    index: int


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

    @abstractmethod
    def describe(self, batch_size: int = 1, num_workers: int = 0) -> dict[str, Any]:
        """Get a description of the dataset, including the number of samples and other relevant information."""


class AnyVisionSegmentationOutput(AnyOutput):
    """Base class for any vision segmentation output.

    It specifies the output format for any vision segmentation dataset.
    It extends the AnyOutput class to include image and mask attributes.
    """

    image: TvImage
    mask: TvMask


class SingleClassVisionSegmentationOutput(AnyVisionSegmentationOutput):
    label: str


class MultiClassVisionSegmentationOutput(AnyVisionSegmentationOutput):
    labels: set[str]


class AnyVisionSegmentationDataset(VisionDataset, AnyDataset):
    """Base class for any vision dataset for the segmentation tasks in images.

    The image and mask are expected to be in the torchvision format with a shape C, H, W for images and masks.

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

    @abstractmethod
    def describe(self, batch_size: int = 1, num_workers: int = 0) -> dict[str, Any]:
        """Get a description of the dataset."""

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


class SingleClassVisionSegmentationDataset(AnyVisionSegmentationDataset):
    """Base class for any vision dataset for the single class per image segmentation tasks in images.

    The image and mask are expected to be in the torchvision format with a shape C, H, W for images and masks.

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
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            override=override,
        )

    @abstractmethod
    def get_raw(self, index: int) -> SingleClassVisionSegmentationOutput:
        """Get the raw data for the given index."""

    def describe(self, batch_size: int = 1, num_workers: int = 0) -> dict[str, Any]:
        """Make a description of the dataset.

        Args:
            batch_size: The batch size to use for the DataLoader.
            num_workers: The number of workers to use for the DataLoader.

        Returns:
            A dictionary containing the description of the dataset, including:
            - name: The name of the dataset class.
            - sample count: The number of samples in the dataset.
            - shapes: A dictionary with the shape as key and the corresponding number of images in the dataset.
            - areas: A dictionary with min, max, mean, and total area of connected components.
            - object count: A dictionary with min, max, mean, and total object count.
            - per label statistics if applicable.
        """
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=lambda x: x,  # some datasets might contain different sizes of images and masks
        )
        global_shape_count: dict[tuple[int, ...], int] = defaultdict(int)
        global_areas = []
        global_object_count = []
        shape_count_per_label: dict[str, dict[tuple[int, ...], int]] = defaultdict(
            lambda: defaultdict(int)
        )
        areas_per_label = defaultdict(list)
        object_count_per_label = defaultdict(list)

        for batch in tqdm(
            dataloader,
            desc=f"Describing dataset {self.__class__.__name__}",
            unit="batch",
        ):
            for sample in batch:
                label = sample["label"]

                shape = tuple(sample["image"].shape)
                global_shape_count[shape] += 1

                shape_count_per_label[label][shape] += 1

                connected_components = extract_connected_components_from_binary_mask(
                    sample["mask"].permute(1, 2, 0).numpy(), min_area=1
                )

                if not connected_components:
                    continue

                areas = [cc.area for cc in connected_components]
                global_areas.extend(areas)
                areas_per_label[label].extend(areas)

                object_count = len(connected_components)
                global_object_count.append(object_count)
                object_count_per_label[label].append(object_count)

        global_description = vision_segmentation_stats(
            shape_counts=global_shape_count,
            areas=global_areas,
            object_counts=global_object_count,
        )
        label_description = {
            label: vision_segmentation_stats(
                shape_counts=shape_count_per_label[label],
                areas=areas,
                object_counts=object_count_per_label[label],
            )
            for label, areas in areas_per_label.items()
        }

        return (
            {"name": self.__class__.__name__, "sample count": len(self)}
            | global_description
            | label_description
        )


class MultiClassVisionSegmentationDataset(AnyVisionSegmentationDataset):
    """Base class for any vision dataset for the multi class per image segmentation tasks in images.

    The image and mask are expected to be in the torchvision format with a shape C, H, W for images and masks.

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
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            override=override,
        )

    @abstractmethod
    def get_raw(self, index: int) -> MultiClassVisionSegmentationOutput:
        """Get the raw data for the given index."""

    def describe(self, batch_size: int = 1, num_workers: int = 0) -> dict[str, Any]:
        raise NotImplementedError(
            "The describe method is not implemented for MultiClassVisionSegmentationDataset. "
        )


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

    def __getitem__(self, index: int) -> AnyOutput:
        """Get the raw data of the dataset for the given index."""
        return self._dataset.get_raw(index)

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self._dataset)
