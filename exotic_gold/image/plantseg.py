from pathlib import Path
from typing import Callable

from exotic_gold.utils.zenodo import ZenodoZipBase


PLANTSEG_VERSIONS = {
    1: {
        "record_id": "13762907",
        "name": "plantseg.zip",
    },
    2: {
        "record_id": "13958858",
        "name": "plantsegv2.zip",
    },
    3: {
        "record_id": "14935094",
        "name": "plantsegv3.zip",
    },
}


class PlantSeg(ZenodoZipBase):
    """
    PlantSeg Dataset from Zenodo.
    """

    def __init__(
        self,
        root: str | Path,
        version: int = 3,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
        override: bool = False,
    ) -> None:
        if version not in PLANTSEG_VERSIONS:
            raise ValueError(
                f"Version {version} is not available. Available versions are {list(PLANTSEG_VERSIONS.keys())}."
            )

        self.version = version
        self.record_id = PLANTSEG_VERSIONS[version]["record_id"]
        self.name = PLANTSEG_VERSIONS[version]["name"]

        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            override=override,
        )
