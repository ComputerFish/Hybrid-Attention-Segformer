"""
This module creates a custom PyTorch Dataset for loading the Mapillary Vistas 
Dataset.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from typing import Tuple, Any, Dict


class MapillaryDataset(Dataset):
    """
    Attributes:
    - Loads the Mapillary config json file to identify the color to category relationship.
    - Process images by normalizing and converting to tensor.
    - Process RGB mask into its class ID tensor.
    """

    def __init__(self, args: Any, mode:str="training"):
        """
        Define the fields used for Dataset operations.

        Args:
            args:       Dataset Configuration from command line arguments.
            mode:       "training" or "validation".

        """
        super().__init__()

        # Field args
        self.config_path = os.path.join(args.dataset_dir, args.config_path)
        self.images_dir = os.path.join(args.dataset_dir, mode, args.image_dir)
        self.mask_dir = os.path.join(args.dataset_dir, mode, args.mask_dir)
        self.image_ext = args.image_ext
        self.mask_ext = args.mask_ext
        self.seed = args.seed

        # Mapillary config json data
        config = self.load_Mapillary_config()
        self.num_classes      = config["num_classes"]
        self.rgb_to_id_dict   = config["rgb_to_id_dict"]
        self.id_to_rgb_dict   = config["id_to_rgb_dict"]
        self.id_to_label_dict = config["id_to_label_dict"]
        self.rgb_to_id_lut    = config["rgb_to_id_lut"]

        # Read off all image ids from directory
        self.image_ids: list[str] = sorted([
            os.path.splitext(file)[0]
            for file in os.listdir(self.images_dir)
            if file.lower().endswith((".png", ".jpg"))
        ])

        # Image transform
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)
        ])
    
    
    def __len__(self):
        """
        Return:
            The number of samples in the dataset.
        """
        return len(self.image_ids)
    

    def __getitem__(self, i:int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Retrieve a single image and mask at index, process it, then return them.

        args:
            i (int): Index of the sample to retrieve.
        
        Returns:
            Tuple[Image tensor, Mask tensor, image_id]
        """
        # Load image and mask
        image_id = self.image_ids[i]
        image_path = os.path.join(self.images_dir, image_id + self.image_ext)
        mask_path = os.path.join(self.mask_dir, image_id + self.mask_ext)
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # Process Image and mask
        image_tensor = self.image_transform(image)
        mask_tensor = self.encode_mask(mask)

        return image_tensor, mask_tensor, image_id
    

    def load_Mapillary_config(self) -> Dict[str, Any]:
        """
        Loads metadata from Mapillary config JSON and return it as dict.

        Returns:
            Dict of:
                num_classes       : number of classes identified by color
                rgb_to_id_dict    : {(R, G, B) -> class_id}
                id_to_rgb_dict    : {class_id -> (R, G, B)}
                id_to_label_dict  : {class_id -> "label name"}
                rgb_to_id_lut     : 256 x 256 x 256 NumPy array for fast vectorized RGB->ID lookup
        """

        # Load the Mapillary config json file
        with open(self.config_path, "r") as file:
            config = json.load(file)

        # Items to initialize, process, then return
        num_classes = len(config["labels"])
        rgb_to_id_dict = {} # {(R,G,B) -> class_id}
        id_to_rgb_dict = {} # {class_id -> (R,G,B)}
        id_to_label_dict = {} # {class_id -> string label}
        rgb_to_id_lut = np.full( # fast lookup table for rgb_to_id
            (256, 256, 256),
            fill_value=num_classes - 1,
            dtype=np.int64
        )

        # Process all items with Mapillary metadata
        for class_id, entry in enumerate(config["labels"]):
            r, g, b = entry["color"]
            rgb_tuple = (r, g, b)
            label_name = entry["name"]

            # Dictonary mappings
            rgb_to_id_dict[rgb_tuple] = class_id
            id_to_rgb_dict[class_id] = rgb_tuple
            id_to_label_dict[class_id] = label_name

            # Vectorized lookup table entry
            rgb_to_id_lut[r, g, b] = class_id

        # Return as dict (to be destructured)
        return {
            "num_classes":      num_classes,
            "rgb_to_id_dict":   rgb_to_id_dict,
            "id_to_rgb_dict":   id_to_rgb_dict,
            "id_to_label_dict": id_to_label_dict,
            "rgb_to_id_lut":    rgb_to_id_lut,
        }
    
    
    def encode_mask(self, mask_rgb: Image.Image) -> torch.Tensor:
        """
        Convert an RGB mask image into a class ID tensor.

        Args:
            mask_rgb:   RGB segmentation mask as a PIL image.

        Returns:
            torch.Tensor: (H, W) tensor of class IDs.
        """
        # convert tensor to np
        mask_np = np.array(mask_rgb)

        # Vectorized lookup convert, no loop
        mask_ids = self.rgb_to_id_lut[
            mask_np[..., 0],
            mask_np[..., 1],
            mask_np[..., 2]
        ]

        return torch.from_numpy(mask_ids).long()

