"""
Real-time semantic segmentation demo using a SegFormer model.

This script loads Mapillary metadata, builds the selected SegFormer variant,
loads pretrained weights, and runs live inference on a webcam feed. Each frame
is preprocessed, passed through the model, converted to a colorized prediction,
and displayed alongside the original image and an overlay visualization. FPS is
computed in real time for performance monitoring.
"""

import os
import sys
import json
import time
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Tuple, Dict

sys.path.append("../trainer")
from demo_config import config_args
from model import (
	Segformer_B0, 
	Segformer_B0_modified_1, 
	Segformer_B0_modified_2, 
	Segformer_B2, 
	Segformer_B2_modified_1
)
from train import mask2D_ids_to_RGB_image


def load_mapillary_metadata(config_filepath:str) -> Tuple[int, Dict]:
    """
    Load Mapillary config JSON for coloring mask.

    args:
        config_filepath: Mapillary config json filepath.

    Returns:
        num_classes
        id_to_rgb_dict: {class_id -> (R, G, B)}
    """
    
    # Load the Mapillary config json file
    with open(config_filepath, "r") as f:
        config = json.load(f)

    # Items to initialize, process, then return
    id_to_rgb_dict = {}
    num_classes = len(config["labels"])

    for class_id, entry in enumerate(config["labels"]):
        r, g, b = entry["color"]
        id_to_rgb_dict[class_id] = (r, g, b)

    return num_classes, id_to_rgb_dict


def main():
    # ------------------------------- Variables -------------------------------

    # Command line arguments defined in the demo_config.py
    args = config_args.parse_args()
    # Maps model name to model class
    model_class_map = {
        "segformer_b0": Segformer_B0,
        "segformer_b0_modified_1": Segformer_B0_modified_1,
        "segformer_b0_modified_2": Segformer_B0_modified_2,
        "segformer_b2": Segformer_B2,
        "segformer_b2_modified_1": Segformer_B2_modified_1,
    }
    # Preprocessing pipeline for camera feed
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ])

    # ------------------------ Load Mapillary Metadata ------------------------

    config_filepath = os.path.join(args.dataset_dir, args.config_path)
    num_classes, id_to_rgb_dict = load_mapillary_metadata(config_filepath)
    print(f"Loaded Mapillary metadata: {num_classes} classes")

    # ------------------------- Build and Load Model -------------------------

    # Select model class based on model name
    model_name = args.model.lower().strip()
    if model_name not in model_class_map:
        raise ValueError(
            f"Invalid model '{args.model}'. "
            f"Available models: {', '.join(model_class_map.keys())}"
        )
    model_class = model_class_map[model_name]

    # Build model
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model_class(num_classes=num_classes)

    # Load model weights
    state = torch.load(args.model_filepath, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    
    print(f"Loaded {args.model} from {args.model_filepath} on {DEVICE}")

    # ---------------------- Live Feed Segmentation Loop ----------------------

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera")
        return
    
    print("Press 'q' to quit.")

    prev_time = time.time() # for fps calculation

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to grab frame")
            break

        # Preprocess frame (BGR -> RGB -> PIL -> Tensor)
        orig_h, orig_w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pil_image = pil_image.resize((args.image_size, args.image_size), 
                                     Image.BILINEAR)
        input_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)

        # Run model inference
        with torch.no_grad():
            logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

        # Convert class IDs to RGB mask
        pred_rgb_pil = mask2D_ids_to_RGB_image(pred, id_to_rgb_dict)
        pred_rgb_np = np.array(pred_rgb_pil)
        pred_bgr = cv2.cvtColor(pred_rgb_np, cv2.COLOR_RGB2BGR)
        pred_bgr = cv2.resize(pred_bgr, (orig_w, orig_h), 
                              interpolation=cv2.INTER_NEAREST)

        # Blend mask and original frame
        overlay = cv2.addWeighted(frame, 0.5, pred_bgr, 0.5, 0)

        # Compute and draw FPS
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(pred_bgr, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Display windows
        cv2.imshow("Original", frame)
        cv2.imshow("Mask Only", pred_bgr)
        cv2.imshow("Overlay", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
