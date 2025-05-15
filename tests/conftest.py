import os


TEST_DATASET_LOADING = os.environ.get("TEST_DATASET_LOADING", "false").lower() in (
    "1",
    "true",
    "yes",
)

ZENODO_API_TOKEN = os.environ.get("ZENODO_API_TOKEN", None)
