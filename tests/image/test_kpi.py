import pytest
from pathlib import Path

import synapseclient
import synapseutils

from any_gold.image.kpi import KPITask1PatchLevel
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

        image, mask, target, index = dataset[0]
        assert index == 0, "Index is not as expected"
        assert image.shape == (1, 3, 2048, 2048), "Image shape is not as expected"
        assert mask.shape == (1, 1, 2048, 2048), "Mask shape is not as expected"
        assert dataset.get_image_path(0) == Path(
            "/storage/ml/kpi_task1/train/train/normal/normal_F3/img/normal_F3_585_13312_22528_img.jpg"
        ), "Image path is not as expected"
        assert target == "normal"
