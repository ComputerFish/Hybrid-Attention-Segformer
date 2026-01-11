"""
This module provides core functions for training, validating, and generating
predictions for a semantic segmentation model. It includes utilities for a
single training epoch, validation with full metric computation, and saving 
prediction outputs for inspection.

Functions:
    train_one_epoch:
        Perform one epoch of model training.

    validate:
        Evaluate the model on a validation set, compute confusion-matrix-based
        metrics (IoU, Dice, pixel accuracy), and return summary statistics.

    mask2D_ids_to_RGB_image:
        Convert a 2D class ID mask into an RGB colorized PIL image.

    predict_mask_and_save:
        Run inference on a dataloader and save original images, masks, 
        and predicted masks to structured output directories.
"""


import os
import shutil
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image
from typing import Dict, Tuple


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> float:
    """
    Train the model for one epoch on the given training dataloader.

    Args:
        model:          Neural network model to train.
        dataloader:     DataLoader providing training batches.
        criterion:      Loss function used for optimization.
        optimizer:      Optimizer for updating model parameters.
        device:         Device to run computations on (e.g., 'cuda' or 'cpu').

    Returns:
        float: Average training loss for the epoch.
    """

    model.train()
    scaler = GradScaler(device)

    # Loop through batches with tqdm progress bar
    running_loss = 0.0
    dataloader_tqdm = tqdm(dataloader, desc="Training", unit="batch", leave=False)
    for images, masks, _ in dataloader_tqdm:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward Pass + loss under autocast
        with autocast(device_type=device.type):
            logits = model(images)
            loss = criterion(logits, masks)
        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        # Update progress bar
        dataloader_tqdm.set_postfix({"loss": f"{loss.item():.3f}"})

    # Calculate and return average loss
    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss


def validate(model, dataloader, criterion, device, num_classes
             ) -> Tuple[float, float, list[float], float, list[float], float]:
    """
    Validate the model on the given validation dataloader.

    Args:
        model:          Neural network model being evaluated.
        dataloader:     DataLoader providing validation batches.
        criterion:      Loss function used for evaluation.
        device:         Device to run computations on.
        num_classes:    Number of segmentation classes.

    Returns:
        tuple:
            avg_loss (float)
            pixel_acc (float)
            per_class_iou (list[float])
            mean_iou (float)
            per_class_dice (list[float])
            mean_dice (float)
    """

    model.eval()
    running_loss = 0.0

    # global confusion matrix accumulated over batches
    confusion_matrix = torch.zeros((num_classes, num_classes), device=device, dtype=torch.float32)

    with torch.no_grad():
        dataloader_tqdm = tqdm(dataloader, desc="Validation", unit="batch", leave=False)
        for images, masks, _ in dataloader_tqdm:
            # Predict masks
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)

            # Update confusion matrix on GPU
            flat_pred = preds.view(-1)
            flat_true = masks.view(-1)

            valid = (flat_true >= 0) & (flat_true < num_classes)
            flat_pred = flat_pred[valid]
            flat_true = flat_true[valid]

            current_confusion_matrix = torch.bincount(
                flat_true * num_classes + flat_pred,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)

            confusion_matrix += current_confusion_matrix

            # Update progress bar
            dataloader_tqdm.set_postfix({"loss": f"{loss.item():.3f}"})

    # Metrics from accumulated confusion matrix
    TP = torch.diag(confusion_matrix)
    FP = confusion_matrix.sum(0) - TP
    FN = confusion_matrix.sum(1) - TP
    denom = TP + FP + FN
    iou = TP / torch.clamp(denom, min=1)
    dice = (2 * TP) / torch.clamp(2 * TP + FP + FN, min=1)
    pixel_acc = TP.sum() / torch.clamp(confusion_matrix.sum(), min=1)
    avg_loss = running_loss / len(dataloader.dataset)

    return (
        avg_loss,
        pixel_acc.item(),
        iou.cpu().tolist(),
        iou.mean().item(),
        dice.cpu().tolist(),
        dice.mean().item(),
    )


def mask2D_ids_to_RGB_image(
        mask2D_ids: np.ndarray, 
        id_to_rgb_dict: Dict[int, Tuple[int, int, int]]
    ) -> Image.Image :
    """
    Convert a 2D class ID mask into an RGB PIL image.

    Args:
        mask2D_ids:      2D NumPy array (H, W) of class IDs.
        id_to_rgb_dict:  Mapping from class_id -> (R, G, B) color.

    Returns:
        Image.Image: RGB colorized mask image.
    """

    h, w = mask2D_ids.shape
    img_array = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert mask of class ids into color
    for class_id, rgb_color in id_to_rgb_dict.items():
        img_array[mask2D_ids == class_id] = rgb_color

    return Image.fromarray(img_array)


def predict_mask_and_save(args, model, dataloader, device, id_to_rgb_dict, out_path):
    """
    Run inference on the dataloader and save original images, masks, and 
    predicted masks to disk.

    Args:
        args:              Runtime configuration and dataset paths.
        model:             Trained segmentation model.
        dataloader:        DataLoader providing validation/test batches.
        device:            Device to run inference on.
        id_to_rgb_dict:    Mapping from class_id -> (R, G, B) for visualization.
        out_path:          Directory where outputs will be written.
    """

    model.eval()

    with torch.no_grad():
        dl_tqdm = tqdm(dataloader, desc="Saving Prediction Images", leave=False)
        for images, masks, image_ids in dl_tqdm: # for every batch
            # Make image predictions for current batch
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()   # [B, H, W]
            masks = masks.numpy()                               # [B, H, W]

            # Loop through image predictions in current batch, save to file
            for i in range(len(image_ids)):
                image_id = image_ids[i]

                # Create directory: ../outputs/<model>/predictions/<image_id>/
                out_dir = os.path.join(out_path, image_id)
                os.makedirs(out_dir, exist_ok=True)

                # Save original input image
                orig_img_path = os.path.join(
                    args.dataset_dir,
                    "validation",
                    args.image_dir,
                    image_id + args.image_ext
                )
                shutil.copyfile(
                    orig_img_path,
                    os.path.join(out_dir, "image" + args.image_ext)
                )

                # Save mask image
                mask_image = mask2D_ids_to_RGB_image(masks[i], id_to_rgb_dict)
                mask_image.save(os.path.join(out_dir, "mask" + args.mask_ext))

                # Save pred image
                pred_image = mask2D_ids_to_RGB_image(preds[i], id_to_rgb_dict)
                pred_image.save(os.path.join(out_dir, "pred" + args.mask_ext))

