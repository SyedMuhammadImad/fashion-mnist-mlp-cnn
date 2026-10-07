"""
AI473 Complex Computing Problem
Contributor: Syed Muhammad Imad
Task: Fashion-MNIST clothing item recognition using MLP and CNN.

Run:
    python f2023376179_code.py

This script downloads Fashion-MNIST, trains both models, evaluates them on the
test set, and saves result images/tables in a generated outputs folder.
"""

import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_loaders():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    train_full = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    val_size = len(train_full) // 10
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(
        train_full, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total
        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    history["train_time_sec"] = time.perf_counter() - start
    return history


def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            predictions = logits.argmax(1).cpu().numpy()
            y_pred.extend(predictions.tolist())
            y_true.extend(y.numpy().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro")),
        "recall": float(recall_score(y_true, y_pred, average="macro")),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": cm.tolist(),
    }


def plot_curves(histories):
    epochs = range(1, EPOCHS + 1)

    plt.figure(figsize=(8, 5))
    for name, history in histories.items():
        plt.plot(epochs, history["train_loss"], marker="o", label=f"{name} train")
        plt.plot(
            epochs,
            history["val_loss"],
            marker="s",
            linestyle="--",
            label=f"{name} val",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and validation loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, history in histories.items():
        plt.plot(epochs, history["train_accuracy"], marker="o", label=f"{name} train")
        plt.plot(
            epochs,
            history["val_accuracy"],
            marker="s",
            linestyle="--",
            label=f"{name} val",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_curves.png", dpi=180)
    plt.close()


def plot_confusion_matrix(cm, model_name):
    cm = np.array(cm)
    plt.figure(figsize=(8, 6))
    image = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    plt.yticks(ticks, CLASS_NAMES, fontsize=8)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{model_name} confusion matrix")

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name.lower()}_confusion_matrix.png", dpi=180)
    plt.close()


def save_results(results, histories):
    rows = []
    for model_name, result in results.items():
        metrics = result["metrics"]
        rows.append(
            {
                "model": model_name,
                "params": result["params"],
                "train_time_sec": round(histories[model_name]["train_time_sec"], 2),
                "test_accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
            }
        )

    with open(OUTPUT_DIR / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "histories": histories}, f, indent=2)

    print("\nFinal comparison:")
    for row in rows:
        print(row)


def main():
    set_seed()
    OUTPUT_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_loaders()
    models = {"MLP": MLP(), "CNN": SmallCNN()}
    histories = {}
    results = {}

    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")
        model = model.to(device)
        history = train_model(model, train_loader, val_loader, device)
        metrics = evaluate_model(model, test_loader, device)
        histories[model_name] = history
        results[model_name] = {
            "params": count_params(model),
            "metrics": metrics,
        }
        plot_confusion_matrix(metrics["confusion_matrix"], model_name)

    plot_curves(histories)
    save_results(results, histories)
    print(f"\nSaved generated outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
