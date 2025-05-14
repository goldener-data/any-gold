import os
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable


from torchvision.datasets import VisionDataset
from zenodo_client import Zenodo

from torchvision.datasets.utils import extract_archive


class ZenodoDataset(VisionDataset):
    """Base class for Zenodo datasets.

    Zenodo is an open repository for research data and software.
    It is accessible at https://zenodo.org/.

    This class specify the basic way to download data from Zenodo. Each inherited class must implement the
    `_move_data_to_root` method to move the data to the root directory after downloading it from Zenodo.
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
        """Make the data available in root directory.

        This method can be used to extract the data from an archive or to reorganise the data after downloading it.
        """

    def download(self) -> None:
        """Download the data from Zenodo and store the dataset in the root folder."""
        ZENODO_API_TOKEN = os.environ.get("ZENODO_API_TOKEN")
        if ZENODO_API_TOKEN is None:
            raise ValueError(
                "Please set the ZENODO_API_TOKEN environment variable to access Zenodo Dataset."
            )

        with TemporaryDirectory() as tmpdir:
            # Download the dataset from Zenodo
            zenodo = Zenodo(access_token=ZENODO_API_TOKEN)
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
    """Base class for Zenodo datasets that are zipped."""

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
