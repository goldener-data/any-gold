# any-gold

Have you ever been in a situation where you wanted to experiment with a new dataset and wasted a few hours
of your time before even having access to the data? We did, and we truly believe that it should not be like that anymore.

Any Gold is thus a comprehensive collection of custom PyTorch Dataset implementations for
publicly available datasets across various modalities.

## Purpose

The goal of this repository is to provide custom PyTorch `Dataset` classes
that are compatible with PyTorch's `DataLoader` to facilitate experimentation
with publicly available datasets. Each dataset implementation includes
automated download functionality to locally cache the data before use. Instead of spending time to access the data,
you can focus on experimenting with it.

## Features

- **PyTorch Integration**: All datasets implement the PyTorch `Dataset` interface
- **Automatic Downloads**: Built-in functionality to download and cache datasets
  locally
- **Multimodal Support**: Datasets spanning various data types and domains
- **Consistent API**: Uniform interface across different dataset implementations
- **Minimal Dependencies**: Core dependencies are managed with `uv`

## Available Datasets

### Image Datasets

- `PlantSeg`: Large-scale in-the-wild dataset for plant disease segmentation ([Paper](https://arxiv.org/abs/2409.04038), [Zenodo](https://zenodo.org/records/14935094))
- `MVTecADDataset`: Anomaly detection dataset for industrial inspection ([Paper](https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf), [Hugging Face](https://huggingface.co/datasets/TheoM55/mvtec_all_objects_split))
- `KPITask1PatchLevel`: A dataset for kidney disease segmentation ([Paper](https://arxiv.org/pdf/2502.07288), [Synapse](https://www.synapse.org/Synapse:syn63688309))
- `DeepGlobeRoadExtraction`: Road extraction from satellite images ([Paper](https://arxiv.org/pdf/1805.06561), [Kaggle](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset))

## Usage

```python
import any_gold as ag
from torch.utils.data import DataLoader

# Initialize dataset (downloads data if not already present)
dataset = ag.AnyDataset()

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through batches
for batch in dataloader:
  # Your training/evaluation code here
  pass
```

## Contributing

### Process

Contributions are welcome! To contribute to this project:

1. Fork the repository on GitHub
2. Clone your fork: `git clone https://github.com/yourusername/any-gold.git`
3. Create a new branch for your feature: `git checkout -b feature-name`
4. Install development dependencies (see below)
5. Set up pre-commit hooks: `uv run pre-commit install`
6. Implement a new class that inherits from `AnyDataset`
7. Include download functionality for the dataset
8. Add appropriate documentation and tests (pytest) for your dataset class
9. Ensure code passes all pre-commit checks
10. Submit a pull request to the main repository

We use pre-commit hooks to maintain code quality:
- Ruff for linting and formatting
- MyPy for type checking


### Installation

Dependencies in this repository are managed with [`uv`](https://github.com/astral-sh/uv),
a fast Python package installer and resolver. The dependencies are defined in the
`pyproject.toml` file.

```bash
# Clone the repository
git clone https://github.com/yourusername/any-gold.git
cd any-gold

# Install dependencies with uv
uv sync --all-extras
source .venv/bin/activate
```
