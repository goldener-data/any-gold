import pytest
from pathlib import Path

from torch.utils.data import RandomSampler, DataLoader

import synapseclient
import synapseutils

from any_gold import KPITask1PatchLevel
from tests.conftest import TEST_DATASET_LOADING, SYNAPSE_API_TOKEN


class TestKPITask1PatchLevel:
    def test_entities(self):
        syn = synapseclient.Synapse()
        syn.login(authToken=SYNAPSE_API_TOKEN)
        walks_train = list(
            synapseutils.walk(syn, KPITask1PatchLevel._ENTITIES["train"]["entity"])
        )
        assert walks_train[0][2][0][1] == "syn60280711"
        walks_val = list(
            synapseutils.walk(syn, KPITask1PatchLevel._ENTITIES["val"]["entity"])
        )
        assert walks_val[0][2][0][1] == "syn60280716"
        walks_test = list(
            synapseutils.walk(syn, KPITask1PatchLevel._ENTITIES["test"]["entity"])
        )
        assert walks_test[0][2][0][1] == "syn63688338"

    @pytest.mark.skipif(
        not TEST_DATASET_LOADING, reason="TEST_DATASET_LOADING is not True"
    )
    def test_dataset(self):
        dataset = KPITask1PatchLevel(
            root=Path("/storage/ml/kpi_task1"),
            split="train",
        )

        assert len(dataset) == 5331, "Dataset length is not as expected"

        output = dataset[0]
        assert output["index"] == 0, "Index is not as expected"
        assert output["image"].shape == (
            3,
            2048,
            2048,
        ), "Image shape is not as expected"
        assert output["mask"].shape == (1, 2048, 2048), "Mask shape is not as expected"
        assert dataset.get_image_path(0) == Path(
            "/storage/ml/kpi_task1/train/train/normal/normal_F3/img/normal_F3_585_13312_22528_img.jpg"
        ), "Image path is not as expected"
        assert output["disease"] == "normal"

        sampler = RandomSampler(dataset, replacement=False, num_samples=5)
        dataloader = DataLoader(
            dataset,
            batch_size=5,
            sampler=sampler,
            num_workers=0,
        )
        for batch in dataloader:
            (
                batch["image"].shape == (5, 3, 1024, 1024),
                "Batch image shape is not as expected",
            )
