import argparse
import csv
import json
from pathlib import Path

import torch

from data import CLASS_NAMES, get_loaders
from evaluate import evaluate_model, top_confusions
from models import MLP, SmallCNN, count_params
from plots import plot_accuracy, plot_confusion_matrix, plot_loss
from train import fit_model, set_seed


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Fashion-MNIST MLP vs CNN experiment")
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument("--model-dir", type=Path, default=root / "models")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-subset", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def write_comparison_csv(path: Path, rows):
    fieldnames = [
        "model",
        "params",
        "train_time_sec",
        "test_accuracy",
        "precision",
        "recall",
        "f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    models = {
        "MLP": MLP(),
        "CNN": SmallCNN(),
    }
    histories = {}
    all_metrics = {
        "metadata": {
            "dataset": "Fashion-MNIST",
            "classes": CLASS_NAMES,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_subset": args.train_subset,
            "validation_fraction": 0.10,
            "seed": args.seed,
            "device": str(device),
        },
        "models": {},
    }
    comparison_rows = []

    for model_name, model in models.items():
        print(f"\n=== {model_name} ===")
        model = model.to(device)
        history = fit_model(model, train_loader, val_loader, device, epochs=args.epochs)
        metrics = evaluate_model(model, test_loader, device)
        params = count_params(model)
        model_file = args.model_dir / f"{model_name.lower()}_fashion_mnist.pt"
        torch.save(model.state_dict(), model_file)

        histories[model_name] = history
        all_metrics["models"][model_name] = {
            "params": params,
            "history": history,
            "test_metrics": metrics,
            "top_confusions": top_confusions(metrics["confusion_matrix"], CLASS_NAMES),
            "model_file": str(model_file.name),
        }

        plot_confusion_matrix(
            metrics["confusion_matrix"],
            CLASS_NAMES,
            model_name,
            args.output_dir / f"{model_name.lower()}_confusion_matrix.png",
        )

        comparison_rows.append(
            {
                "model": model_name,
                "params": params,
                "train_time_sec": round(history["train_time_sec"], 2),
                "test_accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
            }
        )

    plot_loss(histories, args.output_dir / "loss_curves.png")
    plot_accuracy(histories, args.output_dir / "accuracy_curves.png")
    write_comparison_csv(args.output_dir / "comparison.csv", comparison_rows)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("\nComparison")
    for row in comparison_rows:
        print(row)
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
