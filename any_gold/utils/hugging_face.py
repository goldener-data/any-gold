import os
from functools import partial

import datasets
import huggingface_hub.utils
from datasets import load_dataset, DownloadConfig
from datasets.exceptions import DatasetNotFoundError
from datasets import Dataset as HFDataset

from any_gold.utils.dataset import AnyDataset

datasets.logging.set_verbosity_info()


class HuggingFaceDataset(AnyDataset):
    """Base class for Hugging Face datasets.

    Hugging Face is an open repository for machine learning datasets and models.
    It is accessible at https://huggingface.co/.

    This class specifies the basic way to download data from Hugging Face.

    The HUGGINGFACE_API_TOKEN environment variable must be set to access protected Hugging Face datasets.

    Attributes:
        root: The root directory where the dataset is stored.
        path: The path of the dataset on Hugging Face.
        hf_split: The dataset split to use from Hugging Face (e.g., 'train', 'test' but potentially others).
        override: If True, will override the existing dataset in the root directory. Default is False.
        samples: The Hugging Face dataset in torch format.
    """

    def __init__(
        self,
        path: str,
        hf_split: str | None = None,
        override: bool = False,
    ) -> None:
        super().__init__(
            root=huggingface_hub.constants.HUGGINGFACE_HUB_CACHE,
            override=override,
        )

        self.path = path
        self.hf_split = hf_split
        self.samples: HFDataset

        self._setup()

    def _setup(self) -> None:
        """download the data from Hugging face and initialize the dataset in torch format."""
        download_config = DownloadConfig(
            cache_dir=str(self.root), force_download=self.override
        )
        loader = partial(
            load_dataset,
            path=self.path,
            split=self.hf_split,
            download_config=download_config,
        )
        try:
            self.samples = loader().with_format("torch")
        except DatasetNotFoundError:
            HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")
            if HUGGINGFACE_API_TOKEN is None:
                raise ValueError(
                    "HUGGINGFACE_API_TOKEN environment variable must be set."
                )
            download_config.token = HUGGINGFACE_API_TOKEN
            self.samples = loader(token=HUGGINGFACE_API_TOKEN).with_format("torch")
