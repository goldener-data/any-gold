# any-gold

Have you already been in a situation where you wanted to experiment with a new dataset and wasted few hours
of your time before even having access to the data? We did and we truely believe that it should not be like that anymore.

Any Gold is thus comprehensive collection of custom PyTorch Dataset implementations for
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

## Installation

Dependencies in this repository are managed with [`uv`](https://github.com/astral-sh/uv),
a fast Python package installer and resolver. The dependencies are defined in the
`pyproject.toml` file.

```bash
# Clone the repository
git clone https://github.com/yourusername/any-gold.git
cd any-gold

# Install dependencies with uv
uv sync
source .venv/bin/activate
```

## Usage Example

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

Contributions are welcome! To contribute to this project:

1. Fork the repository on GitHub
2. Clone your fork: `git clone https://github.com/yourusername/any-gold.git`
3. Create a new branch for your feature: `git checkout -b feature-name`
4. Install development dependencies: `uv sync --all-extras`
5. Set up pre-commit hooks: `pre-commit install`
6. Implement a new class that inherits from `torch.utils.data.Dataset`
7. Include download functionality for the dataset
8. Add appropriate documentation and usage examples
9. Ensure code passes all pre-commit checks
10. Submit a pull request to the main repository

We use pre-commit hooks to maintain code quality:
- Ruff for linting and formatting
- MyPy for type checking
