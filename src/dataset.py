
"""
dataset.py
PyTorch Dataset and DataLoader factory for the Kaggle COVID-CTset dataset.

Expected dataframe columns:
    filename, patient_id, class

Expected class values:
    covid, normal
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class COVIDCTDataset(Dataset):
    """Load CSV-labelled 16-bit grayscale TIFF images."""

    LABEL_MAP = {
        "normal": 0,
        "covid": 1,
    }

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_paths: dict,
        image_size: int = 224,
        augment: bool = False,
    ):
        required_columns = {"filename", "class"}
        missing = required_columns - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"Dataframe is missing columns: {sorted(missing)}"
            )

        self.df = dataframe.reset_index(drop=True).copy()
        self.image_paths = image_paths
        self.image_size = image_size
        self.augment = augment

        unknown_labels = set(self.df["class"].str.lower()) - set(
            self.LABEL_MAP
        )
        if unknown_labels:
            raise ValueError(f"Unknown labels: {unknown_labels}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        filename = row["filename"]

        # Resolve the TIFF file from the mapping created during data audit.
        paths = self.image_paths.get(filename)
        if not paths:
            raise FileNotFoundError(
                f"No image path found for {filename}"
            )

        with Image.open(paths[0]) as img:
            image = np.asarray(img, dtype=np.float32)

        if image.ndim != 2:
            raise ValueError(
                f"Expected grayscale image, got shape {image.shape}"
            )

        # Robust per-image intensity scaling for the initial baseline.
        low, high = np.percentile(image, [0.5, 99.5])

        if high > low:
            image = np.clip(image, low, high)
            image = (image - low) / (high - low)
        else:
            image = np.zeros_like(image, dtype=np.float32)

        # Tensor shape: [1, H, W]
        image = torch.from_numpy(image.copy()).unsqueeze(0)

        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Training-only augmentation.
        if self.augment:
            if torch.rand(()) < 0.5:
                image = image.flip(-1)

            # Small rotation; fill outside the image with the image minimum.
            angle = (torch.rand(()).item() * 20.0) - 10.0
            theta = torch.tensor(
                [
                    [
                        [np.cos(np.deg2rad(angle)),
                         -np.sin(np.deg2rad(angle)), 0.0],
                        [np.sin(np.deg2rad(angle)),
                         np.cos(np.deg2rad(angle)), 0.0],
                    ]
                ],
                dtype=torch.float32,
            )

            grid = F.affine_grid(
                theta,
                size=(1, 1, self.image_size, self.image_size),
                align_corners=False,
            )
            image = F.grid_sample(
                image.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).squeeze(0)

        # Repeat grayscale input across three channels for the CNN.
        image = image.repeat(3, 1, 1).to(torch.float32)

        mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32
        ).view(3, 1, 1)

        image = (image - mean) / std

        label = self.LABEL_MAP[row["class"].lower()]
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_paths: dict,
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 2,
    device: torch.device | None = None,
):
    """Build separate patient-disjoint train, validation, and test loaders."""

    train_ids = set(train_df["patient_id"])
    val_ids = set(val_df["patient_id"])
    test_ids = set(test_df["patient_id"])

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Patient leakage detected across data splits.")

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    pin_memory = device.type == "cuda"

    train_ds = COVIDCTDataset(
        train_df, image_paths, image_size=image_size, augment=True
    )
    val_ds = COVIDCTDataset(
        val_df, image_paths, image_size=image_size, augment=False
    )
    test_ds = COVIDCTDataset(
        test_df, image_paths, image_size=image_size, augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
