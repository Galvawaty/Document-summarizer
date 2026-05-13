import json
import os

files_to_read = {
    "requirement.txt": "requirement.txt",
    "config.py": "config.py",
    "src/model.py": "src/model.py",
    "src/dataset.py": "src/dataset.py",
    "src/table_detector.py": "src/table_detector.py",
    "src/finetune_indobert.py": "src/finetune_indobert.py",
    "src/inference.py": "src/inference.py"
}

cells = []

# Markdown Header
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# IndoBERT NER Fine-tuning Pipeline\n",
        "Pipeline ini disiapkan untuk dieksekusi di Google Colab. \n",
        "Jalankan cell secara berurutan untuk menyiapkan environment, me-write file pipeline, dan menjalankan training / inference."
    ]
})

# Setup Directory
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 1. Setup Direktori\n",
        "!mkdir -p src scripts data/raw data/processed models/checkpoints output"
    ]
})

for title, path in files_to_read.items():
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # create the file content
    cell_source = [f"%%writefile {title}\n"]
    cell_source.extend([line + "\n" for line in content.split("\n")])
    
    # Clean up the last newline just in case
    if cell_source[-1].endswith("\n\n"):
        cell_source[-1] = cell_source[-1][:-1]
        
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"### Menulis file `{title}`"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cell_source
    })

# Install dependencies
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### Install Dependencies"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!pip install -r requirement.txt"
    ]
})

# Example Usage
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### Jalankan Training"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!python src/finetune_indobert.py --dataset path_ke_dataset.json --epochs 10 --batch 8"
    ]
})

notebook = {
  "cells": cells,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}

with open("pipeline_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
