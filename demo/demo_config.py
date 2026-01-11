"""
This module defines models filepaths and normalization for live demo.
"""

import argparse

config_args = argparse.ArgumentParser()

# ------------------------------ Dataset Configs ------------------------------
config_args.add_argument('--dataset_dir', type = str, default = "../dataset_preprocessed_512x512/", help = "The root directory of the dataset")
config_args.add_argument('--config_path', type = str, default = "config_v1.2.json", help = "JSON Config File")
config_args.add_argument('--model', type = str, default = "Segformer_B2", help = "model name to use in live segmentation")
config_args.add_argument('--model_filepath', type = str, default = "../outputs/Segformer_B2/model.pt", help = "trained model filepath")

# --------------------------- Preprocessing Configs ---------------------------
config_args.add_argument('--mean', nargs=3, type=float, default=[0.485, 0.456, 0.406], help="Normalization mean (RGB)") # Imagenet's RGB mean
config_args.add_argument('--std', nargs=3, type=float, default=[0.229, 0.224, 0.225], help="Normalization std (RGB)") # Imagenet's RGB std
config_args.add_argument('--image_size', type = int, default = 512, help = "OpenCV Window Size (height and width)")