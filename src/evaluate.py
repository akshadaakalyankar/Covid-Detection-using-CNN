
"""
evaluate.py
Evaluation utilities for binary COVID-CT classification.

Label mapping:
    0 = normal
    1 = covid

Expected model output:
    One raw logit per image, trained with sigmoid + BCELoss.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)


CLASS_NAMES = ["normal", "covid"]


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

def get_predictions(model, loader, device):
    """
    Return ground-truth labels and COVID probabilities.

    The model must return raw logits with shape [batch]
    or [batch, 1].
    """
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images).reshape(-1)
            probabilities = torch.sigmoid(logits)

            all_probs.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.reshape(-1).cpu().numpy())

    labels = np.asarray(all_labels, dtype=np.int64)
    probabilities = np.asarray(all_probs, dtype=np.float64)

    return labels, probabilities


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

def plot_confusion_matrix(cm, class_names, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )

    threshold = cm.max() / 2.0 if cm.size else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Confusion matrix saved to {save_path}")


# ---------------------------------------------------------
# ROC curve
# ---------------------------------------------------------

def plot_roc_curve(labels, probabilities, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(np.unique(labels)) < 2:
        print("ROC curve skipped: labels contain only one class.")
        return

    fpr, tpr, _ = roc_curve(labels, probabilities)
    auc = roc_auc_score(labels, probabilities)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
    )
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"ROC curve saved to {save_path}")


# ---------------------------------------------------------
# Training curves
# ---------------------------------------------------------

def plot_training_curves(metrics_path, save_path):
    metrics_path = Path(metrics_path)
    save_path = Path(save_path)

    if not metrics_path.exists():
        print(f"Training metrics not found: {metrics_path}")
        return

    with open(metrics_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    history = data.get("history", data)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    if not any([train_loss, val_loss, train_acc, val_acc]):
        print("No recognized training-history fields found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if train_loss:
        axes[0].plot(train_loss, label="Train Loss")
    if val_loss:
        axes[0].plot(val_loss, label="Validation Loss")

    axes[0].set(
        title="Loss by Epoch",
        xlabel="Epoch",
        ylabel="Loss",
    )
    axes[0].legend()

    if train_acc:
        axes[1].plot(train_acc, label="Train Accuracy")
    if val_acc:
        axes[1].plot(val_acc, label="Validation Accuracy")

    axes[1].set(
        title="Accuracy by Epoch",
        xlabel="Epoch",
        ylabel="Accuracy",
    )
    axes[1].legend()

    for ax in axes:
        ax.grid(alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Training curves saved to {save_path}")


# ---------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------

def evaluate_model(
    model,
    loader,
    device,
    output_dir="results",
    threshold=0.5,
    prefix="test",
):
    """
    Evaluate a trained model on a supplied loader.

    Use validation data to select a threshold.
    Use the test loader for final evaluation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, probabilities = get_predictions(
        model, loader, device
    )

    predictions = (probabilities >= threshold).astype(np.int64)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(
        labels, predictions, zero_division=0
    )
    recall = recall_score(
        labels, predictions, zero_division=0
    )
    f1 = f1_score(
        labels, predictions, zero_division=0
    )

    cm = confusion_matrix(labels, predictions, labels=[0, 1])

    auc = None
    if len(np.unique(labels)) == 2:
        auc = float(roc_auc_score(labels, probabilities))

    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    print(f"\nEvaluation: {prefix}")
    print(f"Threshold:   {threshold:.2f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"COVID recall (sensitivity): {recall:.4f}")
    print(f"COVID F1-score: {f1:.4f}")

    if auc is not None:
        print(f"ROC-AUC:     {auc:.4f}")
    else:
        print("ROC-AUC unavailable: only one class is present.")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    plot_confusion_matrix(
        cm,
        CLASS_NAMES,
        output_dir / f"{prefix}_confusion_matrix.png",
    )

    if auc is not None:
        plot_roc_curve(
            labels,
            probabilities,
            output_dir / f"{prefix}_roc_curve.png",
        )

    results = {
        "split": prefix,
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision_covid": float(precision),
        "recall_covid": float(recall),
        "f1_covid": float(f1),
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "num_images": int(len(labels)),
    }

    with open(
        output_dir / f"{prefix}_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    # Save predictions to support later patient-level analysis.
    np.savez_compressed(
        output_dir / f"{prefix}_predictions.npz",
        labels=labels,
        probabilities=probabilities,
        predictions=predictions,
    )

    print(f"\nMetrics saved in {output_dir}")
    return results
