# Fashion-MNIST CCP Project - F2023376179

This project implements the AI473 Complex Computing Problem for Fashion-MNIST clothing item recognition. It compares a Multi-Layer Perceptron (MLP) and a small Convolutional Neural Network (CNN) using the same train/validation/test pipeline.

## Files

- `src/main.py` - end-to-end training, evaluation, plotting, and metric export.
- `src/data.py` - Fashion-MNIST loading, normalization, and split logic.
- `src/models.py` - MLP and CNN model definitions.
- `outputs/` - generated metrics, comparison table, loss/accuracy plots, and confusion matrices.
- `models/` - saved trained PyTorch model weights.
- `report/f2023376179_CCP_Report.pdf` - final short report.
- `f2023376179_fashion_mnist.ipynb` - notebook version of the experiment.

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the experiment:

```bash
python src/main.py --epochs 5 --train-subset 20000
```

The script downloads Fashion-MNIST automatically through `torchvision.datasets.FashionMNIST`, trains both models, and writes all results into `outputs/`.


## Publication copy

Published 5 October 2026 at the owner's request. This is a sanitized source snapshot. Original local Git history and original files remain unchanged. Pictures, videos, binary archives, private/runtime data, dependency folders and credentials are excluded. Notebook outputs, attachments and incidental metadata are removed. Documents are text-only extracts. Media references and redacted configuration may need replacements before running. No claim of successful rerun, production readiness, sole authorship or independent validation is implied.

Coursework and assisted development artifacts. Publication does not rerun or validate saved training results.
