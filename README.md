# Sea Level Change Portal

## Notebook Setup

Create conda environment that includes notebook dependencies, and make it available to run as a notebook kernel:
```
conda env create -f environment.yml
conda activate slcp_notebook
python -m ipykernel install --user --name=slcp_notebook
```