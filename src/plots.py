from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_loss(history_by_model, output_path: str | Path):
    output_path = Path(output_path)
    plt.figure(figsize=(8, 5))
    for model_name, history in history_by_model.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.plot(epochs, history["train_loss"], marker="o", label=f"{model_name} train")
        plt.plot(epochs, history["val_loss"], marker="s", linestyle="--", label=f"{model_name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and validation loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_accuracy(history_by_model, output_path: str | Path):
    output_path = Path(output_path)
    plt.figure(figsize=(8, 5))
    for model_name, history in history_by_model.items():
        epochs = range(1, len(history["train_accuracy"]) + 1)
        plt.plot(epochs, history["train_accuracy"], marker="o", label=f"{model_name} train")
        plt.plot(epochs, history["val_accuracy"], marker="s", linestyle="--", label=f"{model_name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_confusion_matrix(cm, class_names, model_name: str, output_path: str | Path):
    output_path = Path(output_path)
    cm = np.array(cm)
    plt.figure(figsize=(8, 6))
    image = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right", fontsize=8)
    plt.yticks(ticks, class_names, fontsize=8)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{model_name} confusion matrix")

    threshold = cm.max() / 2.0
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
    plt.savefig(output_path, dpi=180)
    plt.close()
