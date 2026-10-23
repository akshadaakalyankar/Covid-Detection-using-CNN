
"""
train.py
Training utilities for binary COVID-CT classification.

Label mapping:
    0 = normal
    1 = covid

Models return one raw logit per image.
Training uses sigmoid + BCELoss.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score

from models import BaselineCNN


# ---------------------------------------------------------
# Model factory
# ---------------------------------------------------------

def build_model():
    """Create the project's CNN model."""
    return BaselineCNN(num_classes=1)

# ---------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float().reshape(-1)

        optimizer.zero_grad()

        logits = model(images).reshape(-1)
        probabilities = torch.sigmoid(logits)

        loss = criterion(probabilities, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = (probabilities >= 0.5).float()
        correct += (predictions == labels).sum().item()
        total += batch_size

    if total == 0:
        raise ValueError("Training DataLoader is empty.")

    return total_loss / total, correct / total


# ---------------------------------------------------------
# Validate model
# ---------------------------------------------------------

def validate_one_epoch(model, loader, criterion, device):
    """Evaluate loss, accuracy, and ROC-AUC without updating weights."""

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().reshape(-1)

            logits = model(images).reshape(-1)
            probabilities = torch.sigmoid(logits)

            loss = criterion(probabilities, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            predictions = (probabilities >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += batch_size

            all_labels.extend(
                labels.cpu().numpy().astype(np.int64)
            )
            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    if total == 0:
        raise ValueError("Validation DataLoader is empty.")

    labels_array = np.asarray(all_labels)
    probabilities_array = np.asarray(all_probabilities)

    auc = None
    if len(np.unique(labels_array)) == 2:
        auc = float(
            roc_auc_score(labels_array, probabilities_array)
        )

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "auc": auc,
    }


# ---------------------------------------------------------
# Full training loop
# ---------------------------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=20,
    learning_rate=1e-3,
    output_dir="results",
    model_name="baseline",
):
    """Train a model and save its best validation-AUC checkpoint."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # BCE requires probabilities and labels with matching shapes.
    criterion = nn.BCELoss()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise ValueError("The model has no trainable parameters.")

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
    }

    best_auc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0

    checkpoint_path = output_dir / f"{model_name}_best.pth"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["auc"])

        auc_display = (
            f"{val_metrics['auc']:.4f}"
            if val_metrics["auc"] is not None
            else "N/A"
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val AUC: {auc_display}"
        )

        # Prefer ROC-AUC for checkpoint selection because it evaluates
        # ranking quality across thresholds.
        if val_metrics["auc"] is not None:
            improved = val_metrics["auc"] > best_auc
        else:
            improved = val_metrics["loss"] < best_val_loss

        if improved:
            best_auc = (
                val_metrics["auc"]
                if val_metrics["auc"] is not None
                else best_auc
            )
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch

            # Save only the state dict for compatibility with evaluate.py.
            torch.save(model.state_dict(), checkpoint_path)

    metrics = {
        "model": model_name,
        "best_epoch": best_epoch,
        "best_val_auc": (
            best_auc if best_auc >= 0 else None
        ),
        "best_val_loss": best_val_loss,
        "history": history,
    }

    metrics_path = output_dir / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"Training history saved to: {metrics_path}")

    return model, history
