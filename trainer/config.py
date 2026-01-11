"""
This module defines hyperparameters and command-line arguments for model 
training.

It uses Python's `argparse.ArgumentParser` to provide a structured interface for 
configuring runtime parameters such as training epochs, device_name, and 
other customizable options. These arguments are parsed from the command line and 
used throughout the training pipeline.

Example:
    python3 main.py --model Segformer_B0
"""

import argparse

config_args = argparse.ArgumentParser()

# ------------------------------ Dataset Configs ------------------------------
config_args.add_argument('--dataset_dir', type = str, default = "../dataset_preprocessed_512x512/", help = "The root directory of the dataset")
config_args.add_argument('--config_path', type = str, default = "config_v1.2.json", help = "JSON Config File")
config_args.add_argument('--image_dir', type = str, default = "images", help = "The root directory of the image files")
config_args.add_argument('--mask_dir', type = str, default = "labels", help = "The root directory of the mask files")
config_args.add_argument('--image_ext', type = str, default = ".jpg", help = "File extension of images")
config_args.add_argument('--mask_ext', type = str, default = ".png", help = "File extension of mask")

# --------------------------- Preprocessing Configs ---------------------------
config_args.add_argument('--seed', type = int, default = 42, help = "seed for reproduciability")
config_args.add_argument('--mean', nargs=3, type=float, default=[0.485, 0.456, 0.406], help="Normalization mean (RGB)") # Imagenet's RGB mean
config_args.add_argument('--std', nargs=3, type=float, default=[0.229, 0.224, 0.225], help="Normalization std (RGB)") # Imagenet's RGB std

# ------------------------------ Training Configs ------------------------------
config_args.add_argument('--epochs', type = int, default = 20, help = "# of epochs")
config_args.add_argument('--batch', type = int, default = 8, help = "batch size")
config_args.add_argument('--lr', type = float, default = 1e-4, help = "learning rate")
config_args.add_argument('--output_dir', type = str, default = "../outputs", help = "The root directory of the outputs")
config_args.add_argument('--device_name', type = str, default = "0", help = "The available gpu in the cluster, check with nvidia_smi")
config_args.add_argument('--model', type = str, default = "", help = "model name to train Segformer_B0, Segformer_B0_modified, Segformer_B2, etc.")
config_args.add_argument('--version', type = str, default = "", help = "version or modification of current model")
