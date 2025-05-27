import os
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from logging import getLogger

from torchvision.datasets.utils import extract_archive

from zenodo_client import Zenodo

from any_gold.utils.dataset import AnyDataset

logger = getLogger(__name__)


class ZenodoDataset(AnyDataset):
    """Base class for Zenodo datasets.

    Zenodo is an open repository for research data and software.
    It is accessible at https://zenodo.org/.

    This class specifies the basic way to download data from Zenodo. Each inherited class must implement the
    `_move_data_to_root` and `_setup` methods to download the data from Zenodo.

    The ZENODO_API_TOKEN environment variable must be set to access Zenodo datasets.

    Attributes:
        root: The root directory where the dataset is stored.
        record_id: The record ID of the dataset on Zenodo.
        name: The name of the dataset on Zenodo.
        override: If True, will override the existing dataset in the root directory. Default is False.
    """

    def __init__(
        self,
        root: str | Path,
        record_id: str,
        name: str,
        override: bool = False,
    ) -> None:
        super().__init__(root=root, override=override)

        self.record_id = record_id
        self.name = name

        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """download the data from Zenodo and initialize the elements of the dataset."""

    @abstractmethod
    def _move_data_to_root(self, file: Path) -> None:
        """Make the data available in the root directory.

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
            logger.info(f"Downloading {self.record_id} from Zenodo to {tmpdir}")
            file = zenodo.download_latest(
                self.record_id,
                name=self.name,
                force=True,
                parts=[str(tmpdir)],
            )
            if not self.root.exists():
                self.root.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading {self.record_id} from Zenodo to {tmpdir}")
            self._move_data_to_root(file)


class ZenodoZipBase(ZenodoDataset):
    """Base class for Zenodo datasets that are zipped."""

    def __init__(
        self,
        root: str | Path,
        record_id: str,
        name: str,
        override: bool = False,
    ) -> None:
        super().__init__(root, record_id, name, override)

    def _move_data_to_root(self, file: Path) -> None:
        """Move the data to the root directory."""
        extract_archive(file, str(self.root))
