from abc import abstractmethod
from pathlib import Path
from typing import Callable, Any

from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset


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
    def get_raw(self, index: int) -> tuple[Any, ...]:
        """Get the raw data for the given index."""


class AnyVisionDataset(VisionDataset, AnyDataset):
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
