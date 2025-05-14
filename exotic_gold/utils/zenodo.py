import os
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable


from torchvision.datasets import VisionDataset
from zenodo_client import Zenodo

from torchvision.datasets.utils import extract_archive


class ZenodoDataset(VisionDataset):
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
        )
        self.root = root if isinstance(root, Path) else Path(root)
        self.transform = transform
        self.target_transform = target_transform

        self.record_id: str
        self.name: str

        if override or not self.root.exists():
            self.download()

    @abstractmethod
    def _move_data_to_root(self, file: Path) -> None:
        """Make the data available in root directory."""

    def download(self) -> None:
        ZENODO_API_TOKEN = os.environ.get("ZENODO_API_TOKEN")
        if ZENODO_API_TOKEN is None:
            raise ValueError(
                "Please set the ZENODO_API_TOKEN environment variable to access Zenodo Dataset."
            )

        with TemporaryDirectory() as tmpdir:
            # Download the dataset from Zenodo
            zenodo = Zenodo()
            file = zenodo.download_latest(
                self.record_id,
                name=self.name,
                force=True,
                parts=[str(tmpdir)],
            )
            if not self.root.exists():
                self.root.mkdir(parents=True, exist_ok=True)

            self._move_data_to_root(file)


class ZenodoZipBase(ZenodoDataset):
    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
        override: bool = False,
    ) -> None:
        super().__init__(root, transform, target_transform, transforms, override)

    def _move_data_to_root(self, file: Path) -> None:
        """Move the data to the root directory."""
        extract_archive(file, str(self.root))
