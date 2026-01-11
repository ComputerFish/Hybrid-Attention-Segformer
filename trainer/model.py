"""
This module defines multiple SegFormer-based semantic segmentation models. It 
includes the original SegFormer B0 and B2 architectures adapted to a custom 
number of classes, along with modified variants that incorporate additional 
encoder layers and local window-based self-attention.

Models:
    Segformer_B0
    Segformer_B0_modified_1
    Segformer_B0_modified_2
    Segformer_B2
    Segformer_B2_modified_1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import copy
from windowedLocalSelfAttention import WindowLocalSelfAttention


class Segformer_B0(nn.Module):
    """
    SegFormer B0 model adapted for custom semantic segmentation. Loads the 
    pretrained NVIDIA SegFormer-B0 weights and replaces the classifier head 
    to match the required number of classes in our dataset.
    """

    def __init__(self, num_classes: int):
        """
        Initialize the SegFormer B0 model and adjust it for the given number
        of segmentation classes.

        Args:
            num_classes: Number of output segmentation classes.
        """

        super().__init__()

        # Download model from Hugging Face
        hugging_face_segformer = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            hugging_face_segformer,
            ignore_mismatched_sizes=True,
        )

        # Disable extra outputs (saves LOTS of GPU memory)
        self.model.config.output_hidden_states = False
        self.model.config.output_attentions = False
        self.model.config.use_cache = False

        # Replace classifier head to match our num_classes
        # by default cityscapes has 19 classes, we have different
        in_channels = self.model.decode_head.classifier.in_channels
        self.model.decode_head.classifier = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=1,
            bias=True,
        )

        # Init new head weights
        nn.init.xavier_uniform_(self.model.decode_head.classifier.weight)
        if self.model.decode_head.classifier.bias is not None:
            nn.init.zeros_(self.model.decode_head.classifier.bias)

        # Save num_classes for later
        self.num_classes = num_classes


    def forward(self, x:torch.Tensor):
        """
        Forward pass through the model.

        Args:
            x: Input tensor of images (B, C, H, W).

        Returns:
            torch.Tensor: Output logits resized to the original input resolution.
        """

        # Forward pass through Segformer
        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        # Upsample prediction back to input size
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits


class Segformer_B0_modified_1(nn.Module):
    """
    SegFormer B0 variant with a single attention block replaced by a local
    window-based self-attention module. All other components follow the
    standard SegFormer-B0 architecture.
    """
    
    def __init__(self, num_classes: int):
        """
        Initialize the modified SegFormer B0 model. Loads pretrained weights
        and replaces one encoder attention block with local self-attention.

        Args:
            num_classes: Number of output segmentation classes.
        """

        super().__init__()

        # Download model from Hugging Face
        hugging_face_segformer = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            hugging_face_segformer,
            ignore_mismatched_sizes=True,
        )

        # Disable extra outputs (saves LOTS of GPU memory)
        self.model.config.output_hidden_states = False
        self.model.config.output_attentions = False
        self.model.config.use_cache = False

        # Grab encoder, peek into blocks, and replace ONE attention module
        encoder = self.model.segformer.encoder
        stage_index = 1
        block_index = 0 

        # Get that block
        target_block = encoder.block[stage_index][block_index]
        dim = self.model.config.hidden_sizes[stage_index]
        num_heads = self.model.config.num_attention_heads[stage_index]

        # Create a local attention module with the same heads
        self.local_attn = WindowLocalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=7,
        )

        # Replace the attention in that block with local attention
        original_self_attn = target_block.attention.self

        # Monkey patch the forward of the attention
        def new_self_forward(self_attn, hidden_states, height, width, output_attentions=False):
            out = self.local_attn(hidden_states, H=height, W=width)

            if output_attentions:
                return (out, None)
            
            return (out,)

        # Store reference to original forward just incase we need it
        original_self_attn._original_forward = original_self_attn.forward
        original_self_attn.forward = new_self_forward.__get__(original_self_attn, original_self_attn.__class__)

        # Replace classifier head to match our num_classes
        # by default cityscapes has 19 classes, we have different
        in_channels = self.model.decode_head.classifier.in_channels
        self.model.decode_head.classifier = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=1,
            bias=True,
        )

        # Init new head weights
        nn.init.xavier_uniform_(self.model.decode_head.classifier.weight)
        if self.model.decode_head.classifier.bias is not None:
            nn.init.zeros_(self.model.decode_head.classifier.bias)

        # Save num_classes for later
        self.num_classes = num_classes


    def forward(self, x:torch.Tensor):
        """
        Forward pass through the modified SegFormer B0.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            torch.Tensor: Logits upsampled to input image resolution.
        """

        # Forward pass through Segformer
        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits


class Segformer_B0_modified_2(nn.Module):
    """
    SegFormer B0 variant that expands stages 2 and 4 with additional layers
    and replaces all attention modules in those stages with local window-based
    self-attention. Designed to increase model capacity and incorporate more
    localized spatial inductive bias.
    """
    
    def __init__(self, num_classes: int):
        """
        Initialize the modified SegFormer B0 model. Adds additional transformer
        blocks to specific encoder stages and replaces all attention mechanisms
        in those stages with local self-attention.

        Args:
            num_classes: Number of output segmentation classes.
        """

        super().__init__()

        # Download model from Hugging Face
        hugging_face_segformer = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            hugging_face_segformer,
            ignore_mismatched_sizes=True,
        )

        # Disable extra outputs (saves LOTS of GPU memory)
        self.model.config.output_hidden_states = False
        self.model.config.output_attentions = False
        self.model.config.use_cache = False

        # Add more layers to stages 2 and 4
        encoder = self.model.segformer.encoder
        stages_to_expand = [1, 3]  # stages 2 and 4
        num_extra_layers = 2
        
        # Copy layers we need to modify
        for stage_index in stages_to_expand:
            template_layer = encoder.block[stage_index][-1]
            for _ in range(num_extra_layers):
                new_layer = copy.deepcopy(template_layer)
                encoder.block[stage_index].append(new_layer)

        # Replace ALL attention in stages 2 and 4 with local attention
        self.local_attn_modules = nn.ModuleDict()
        for stage_index in stages_to_expand:
            dim = self.model.config.hidden_sizes[stage_index]
            num_heads = self.model.config.num_attention_heads[stage_index]
            
            # Create one local attention module per stage
            local_attn = WindowLocalSelfAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=7,
            )
            self.local_attn_modules[f'stage_{stage_index}'] = local_attn
            
            # Loop through ALL blocks in this stage
            for block_index, block in enumerate(encoder.block[stage_index]):
                original_self_attn = block.attention.self
                
                # Create closure that captures the local_attn for this stage
                def make_new_forward(local_attention_module):
                    def new_self_forward(hidden_states, height, width, output_attentions=False):
                        out = local_attention_module(hidden_states, H=height, W=width)
                        if output_attentions:
                            return (out, None)
                        return (out,)
                    return new_self_forward
                
                # Replace the forward method
                original_self_attn._original_forward = original_self_attn.forward
                original_self_attn.forward = make_new_forward(local_attn)

        # Replace classifier head to match our num_classes
        in_channels = self.model.decode_head.classifier.in_channels
        self.model.decode_head.classifier = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=1,
            bias=True,
        )

        # Init new head weights
        nn.init.xavier_uniform_(self.model.decode_head.classifier.weight)
        if self.model.decode_head.classifier.bias is not None:
            nn.init.zeros_(self.model.decode_head.classifier.bias)

        self.num_classes = num_classes


    def forward(self, x:torch.Tensor):
        """
        Forward pass through the modified SegFormer B0.

        Args:
            x (torch.Tensor): Input batch of RGB images (B, 3, H, W).

        Returns:
            torch.Tensor: Logits upsampled to the original input resolution.
        """

        # Forward pass through Segformer
        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits


class Segformer_B2(nn.Module):
    """
    SegFormer B2 model adapted for custom semantic segmentation. Loads the 
    pretrained NVIDIA SegFormer-B2 weights and replaces the classifier head to 
    match the required number of classes in our dataset.
    """
    
    def __init__(self, num_classes: int):
        """
        Initialize the SegFormer B2 model and adjust it for the given number
        of segmentation classes.

        Args:
            num_classes: Number of output segmentation classes.
        """

        super().__init__()

        # Download model from Hugging Face
        hugging_face_segformer = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            hugging_face_segformer,
            ignore_mismatched_sizes=True,
        )

        # Disable extra outputs (saves LOTS of GPU memory)
        self.model.config.output_hidden_states = False
        self.model.config.output_attentions = False
        self.model.config.use_cache = False

        # Replace classifier head to match our num_classes
        # by default cityscapes has 19 classes, we have different
        in_channels = self.model.decode_head.classifier.in_channels
        self.model.decode_head.classifier = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=1,
            bias=True,
        )

        # Init new head
        nn.init.xavier_uniform_(self.model.decode_head.classifier.weight)
        if self.model.decode_head.classifier.bias is not None:
            nn.init.zeros_(self.model.decode_head.classifier.bias)

        self.num_classes = num_classes


    def forward(self, x: torch.Tensor):
        """
        Forward pass through the SegFormer B2 model.

        Args:
            x (torch.Tensor): Input batch of RGB images (B, 3, H, W).

        Returns:
            torch.Tensor: Logits upsampled to the original input resolution.
        """

        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        # Upsample back to input size
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits


class Segformer_B2_modified_1(nn.Module):
    """
    SegFormer B2 variant that expands stages 2 and 4 with additional layers
    and replaces all attention modules in those stages with local window-based
    self-attention. This increases model capacity and adds localized inductive
    bias to deeper encoder stages.
    """

    def __init__(self, num_classes: int):
        """
        Initialize the modified SegFormer B2 model. Adds additional transformer
        blocks to specific encoder stages and replaces their attention modules
        with local self-attention.

        Args:
            num_classes: Number of output segmentation classes.
        """

        super().__init__()

        # Download model from Hugging Face
        hugging_face_segformer = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            hugging_face_segformer,
            ignore_mismatched_sizes=True,
        )

        # Disable extra outputs for memory savings
        self.model.config.output_hidden_states = False
        self.model.config.output_attentions = False
        self.model.config.use_cache = False

        # Add more layers to stages 2 and 4
        encoder = self.model.segformer.encoder
        stages_to_expand = [1, 3]  # stages 2 and 4
        num_extra_layers = 2
        
        for stage_index in stages_to_expand:
            template_layer = encoder.block[stage_index][-1]
            for _ in range(num_extra_layers):
                new_layer = copy.deepcopy(template_layer)
                encoder.block[stage_index].append(new_layer)

        # Replace ALL attention in stages 2 and 4 with local attention
        self.local_attn_modules = nn.ModuleDict()
        for stage_index in stages_to_expand:
            dim = self.model.config.hidden_sizes[stage_index]
            num_heads = self.model.config.num_attention_heads[stage_index]
            
            # Create one local attention module per stage
            local_attn = WindowLocalSelfAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=7,
            )
            self.local_attn_modules[f'stage_{stage_index}'] = local_attn
            
            # Loop through ALL blocks in this stage
            for block_index, block in enumerate(encoder.block[stage_index]):
                original_self_attn = block.attention.self
                
                # Create closure that captures the local_attn for this stage
                def make_new_forward(local_attention_module):
                    def new_self_forward(hidden_states, height, width, output_attentions=False):
                        out = local_attention_module(hidden_states, H=height, W=width)
                        if output_attentions:
                            return (out, None)
                        return (out,)
                    return new_self_forward
                
                # Replace the forward method
                original_self_attn._original_forward = original_self_attn.forward
                original_self_attn.forward = make_new_forward(local_attn)


        # Replace classifier head to match our num_classes
        in_channels = self.model.decode_head.classifier.in_channels
        self.model.decode_head.classifier = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=1,
            bias=True,
        )

        # Init new head weights
        nn.init.xavier_uniform_(self.model.decode_head.classifier.weight)
        if self.model.decode_head.classifier.bias is not None:
            nn.init.zeros_(self.model.decode_head.classifier.bias)

        self.num_classes = num_classes


    def forward(self, x:torch.Tensor):
        """
        Forward pass through the modified SegFormer B2.

        Args:
            x (torch.Tensor): Input batch of RGB images (B, 3, H, W).

        Returns:
            torch.Tensor: Logits resized to the original input spatial resolution.
        """

        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        # Upsample back to input size
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits
