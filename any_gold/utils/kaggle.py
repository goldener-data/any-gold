from abc import abstractmethod
from logging import getLogger
from pathlib import Path

import kagglehub

from any_gold.utils.dataset import AnyDataset

logger = getLogger(__name__)


class KaggleDataset(AnyDataset):
    """Base class for Kaggle datasets.

    Kaggle is a platform for machine learning competitions and some datasets are made accessible for those competitions.
    It is accessible at https://huggingface.co/.

    This class specifies the basic way to download data from Kaggle.

    Attributes:
        root: The root directory where the dataset is stored.
        handle: The name of the dataset on Kaggle (same as the Kaggle dataset URL).
        override: If True, will override the existing dataset in the root directory. Default is False.
    """

    def __init__(
        self,
        root: str | Path,
        handle: str,
        override: bool = False,
    ) -> None:
        super().__init__(root=root, override=override)

        self.handle = handle

        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """download the data from Kaggle and initialize the elements of the dataset."""

    @abstractmethod
    def _move_data_to_root(self, dataset_directory: Path) -> None:
        """Make the data available in the root directory.

        This method can be used to extract the data from an archive or to reorganise the data after downloading it.
        """

    def download(self) -> None:
        """Download the data from kaggle and store the dataset in the root folder."""

        logger.info(f"Downloading {self.handle} to kagglehub cache.")
        kaggle_cache = kagglehub.dataset_download(
            self.handle,
        )
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)

        logger.info(f"Move {self.handle} from {kaggle_cache} to {self.root}")
        self._move_data_to_root(Path(kaggle_cache))
