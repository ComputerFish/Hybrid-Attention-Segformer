"""
This module performs training and validation of semantic segmentation model for 
the Mapillary Vistas Dataset. The Mapillary Vistas dataset contains street-level 
images and annotations of 65 different categories. This module will manage 
data loading, training loops, model evaluation, logging of metrics, and 
saving mask predictions.

The training pipeline includes:
    - Reproducible seeding
    - Model training and validation per epoch
    - Metrics logging and saving for each epoch
    - Mask Prediction and saving at the very end

How to run:
    python3 main.py --model Segformer_B0
	python3 main.py --model Segformer_B0_modified_1
	python3 main.py --model Segformer_B0_modified_2
	python3 main.py --model Segformer_B2
	python3 main.py --model Segformer_B2_modified_1
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any

from dataset import MapillaryDataset
from train import train_one_epoch, validate, predict_mask_and_save
from config import config_args
from model import (
	Segformer_B0, 
	Segformer_B0_modified_1, 
	Segformer_B0_modified_2, 
	Segformer_B2, 
	Segformer_B2_modified_1
)


def run_training(model_class: type[nn.Module], args: Any):
    """
    Train the given model_class with the given parameter in args, save the 
    trained model, record its metrics, and save its mask predictions.

    Args:
        model_class:    Reference to the model class to train.
        args:           Object containing configuration parameters such as 
                        dataset directories and image settings.
    """

    # ---------------------------- Initialization ----------------------------

    # Restrict visible GPUs and set the target device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_name
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = model_class.__name__
    model_name += "_" + args.version if args.version != "" else ""
    print(f"{model_name} is running on {DEVICE}...\n")

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load train and validation dataset from file
    train_ds = MapillaryDataset(args, mode="training")
    val_ds = MapillaryDataset(args, mode="validation")

    # Initialize dataloaders
    num_workers = max(1, os.cpu_count() - 1) # Optimizing CPU usage
    prefetch_factor = 4 if num_workers > 0 else None
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, 
                          num_workers=num_workers, persistent_workers=True, 
                          prefetch_factor=prefetch_factor, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, 
                        num_workers=num_workers, persistent_workers=True, 
                        prefetch_factor=prefetch_factor, pin_memory=True)
    
    # Initialize model, loss, and optimizer
    model = model_class(num_classes=train_ds.num_classes).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=65) # ignore index 65 because it is "Unlabeled" category
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ----------------------------- Training Loop -----------------------------
    
    log_records = []

    for epoch in range(args.epochs):
        # Train and validate
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_IoU, val_mIoU, val_dice, val_mDice = validate(
            model, val_dl, criterion, DEVICE, train_ds.num_classes)

        # Print metrics
        print(
            f"Epoch [{epoch + 1:02d}/{args.epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Pixel Acc: {val_acc:.4f} | "
            f"mIoU: {val_mIoU:.4f} | "
            f"mDICE: {val_mDice:.4f}"
        )

        # Append to save metrics 
        log_records.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pixel_acc": val_acc,
            "val_mIoU": val_mIoU,
            "val_mDice": val_mDice
        })

    # -------------------- Save logs and predicted images --------------------

    # Create filepath hierarchy
    out_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save training logs
    log_df = pd.DataFrame(log_records)
    log_path = os.path.join(out_dir, "training_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_df.to_csv(log_path, index=False)
    print(f"\nTraining Log is saved to {log_path}")

    # Save model
    model_path = os.path.join(out_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model is saved to {model_path}")

    # Save predictions
    pred_path = os.path.join(out_dir, "predictions")
    predict_mask_and_save(args, model, val_dl, DEVICE, val_ds.id_to_rgb_dict, pred_path)
    print(f"Prediction images are saved {pred_path}")


if __name__ == "__main__":
    args = config_args.parse_args()
    model_class_map = {
        "segformer_b0": Segformer_B0,
        "segformer_b0_modified_1": Segformer_B0_modified_1,
        "segformer_b0_modified_2": Segformer_B0_modified_2,
        "segformer_b2": Segformer_B2,
        "segformer_b2_modified_1": Segformer_B2_modified_1,
    }
    
    # ----------------------------- Guard Clause -----------------------------
    model_name = args.model.lower().strip()

    if model_name == "":
        raise ValueError(
            "--model must be specified. "
            f"Available models: {', '.join(model_class_map.keys())}"
        )

    if model_name not in model_class_map:
        raise ValueError(
            f"Invalid model '{args.model}'. "
            f"Available models: {', '.join(model_class_map.keys())}"
        )

    # ------------------------------- Main Code -------------------------------
    model_class = model_class_map[model_name]
    run_training(model_class, args)