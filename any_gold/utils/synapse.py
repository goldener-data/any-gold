import os
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from logging import getLogger

from torchvision.datasets.utils import extract_archive

import synapseclient
import synapseutils

from any_gold.utils.dataset import AnyDataset

logger = getLogger(__name__)


class SynapseDataset(AnyDataset):
    """Base class for Synapse datasets.

    Synapse is an open repository for biomedical research data and software.
    It is accessible at https://www.synapse.org/.

    This class specifies the basic way to download data from Synapse. Each inherited class must implement the
    `_move_data_to_root` and `_setup` methods to download the data from Synapse.

    The SYNAPSE_API_TOKEN environment variable must be set to access Synapse datasets.

    Attributes:
        root: The root directory where the dataset is stored.
        override: If True, will override the existing dataset in the root directory. Default is False.
        entity: The Synapse entity ID of the dataset.
    """

    def __init__(
        self,
        root: str | Path,
        entity: str,
        override: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            override=override,
        )

        self.entity = entity

        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """download the data from Synapse and initialize the elements of the dataset."""

    @abstractmethod
    def _move_data_to_root(self, files: list[synapseclient.entity.File]) -> None:
        """Make the data available in the root directory.

        This method can be used to extract the data from an archive or to reorganise the data after downloading it.
        """

    def download(self) -> None:
        """Download the data from Synapse and store the dataset in the root folder."""
        SYNAPSE_API_TOKEN = os.environ.get("SYNAPSE_API_TOKEN")
        if SYNAPSE_API_TOKEN is None:
            raise ValueError(
                "Please set the SYNAPSE_API_TOKEN environment variable to access Synapse Dataset."
            )

        with TemporaryDirectory() as tmpdir:
            syn = synapseclient.Synapse()
            syn.login(authToken=SYNAPSE_API_TOKEN)

            logger.info(f"Downloading {self.entity} from Synapse to {tmpdir}")
            files = synapseutils.syncFromSynapse(syn, self.entity, str(tmpdir))

            logger.info(f"Moving files to {self.root}")
            self._move_data_to_root(files)


class SynapseZipBase(SynapseDataset):
    """Base class for Synapse datasets that are zipped."""

    def __init__(
        self,
        root: str | Path,
        entity: str,
        override: bool = False,
    ) -> None:
        super().__init__(root, entity, override)

    def _move_data_to_root(self, files: list[synapseclient.entity.File]) -> None:
        """Move the data to the root directory."""
        for file in files:
            extract_archive(file.path, str(self.root))
