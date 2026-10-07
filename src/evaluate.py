import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@torch.no_grad()
def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        predictions = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(predictions.tolist())
        y_true.extend(y.numpy().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def top_confusions(cm, class_names, top_n: int = 5):
    matrix = np.array(cm, dtype=int).copy()
    np.fill_diagonal(matrix, 0)
    pairs = []
    for true_idx, pred_idx in zip(*np.unravel_index(np.argsort(matrix.ravel())[::-1], matrix.shape)):
        count = int(matrix[true_idx, pred_idx])
        if count == 0:
            break
        pairs.append(
            {
                "true": class_names[true_idx],
                "predicted": class_names[pred_idx],
                "count": count,
            }
        )
        if len(pairs) == top_n:
            break
    return pairs
