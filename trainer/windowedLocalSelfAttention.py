"""
This module implements a local window-based self-attention mechanism used as a 
drop-in replacement for global attention blocks in SegFormer models. It performs 
multi-head self-attention over non-overlapping spatial windows, allowing the 
model to capture fine-grained local context with reduced computational cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowLocalSelfAttention(nn.Module):
    """
    Local window-based multi-head self-attention module.

    This layer performs self-attention within non-overlapping spatial
    windows instead of across the full feature map, reducing computation
    while preserving local spatial context. The input is expected in
    flattened sequence format [B, N, C] where N = H * W.
    """

    def __init__(self, dim, num_heads=4, window_size=7):
        """
        Initialize the local window self-attention module.

        Args:
            dim:          Total embedding dimension (C).
            num_heads:    Number of attention heads.
            window_size:  Size of the non-overlapping attention windows.
        """

        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        # Project input to Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        # Output projection
        self.proj = nn.Linear(dim, dim)


    def forward(self, x, H, W):
        """
        Apply local window-based self-attention.

        Args:
            x (torch.Tensor): Flattened input sequence of shape [B, N, C],
                              where N = H * W.
            H (int):          Original feature map height.
            W (int):          Original feature map width.

        Returns:
            torch.Tensor: Output tensor of shape [B, N, C] after local
                          self-attention within non-overlapping windows.
        """

        B, N, C = x.shape

        # Reshape to 2D grid
        x = x.reshape(B, H, W, C)  # [B, H, W, C]

        # Pad so H and W are multiples of window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            Hp, Wp = H + pad_h, W + pad_w
        else:
            Hp, Wp = H, W

        # Split into windows
        # [B, num_windows_h, num_windows_w, window, window, C]
        num_windows_h = Hp // self.window_size
        num_windows_w = Wp // self.window_size

        # [B, Nh, Wh, Nw, Ww, C]
        x = x.reshape(
            B,
            num_windows_h,
            self.window_size,
            num_windows_w,
            self.window_size,
            C,
        )

        # Move windows to batch dimension
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(-1, self.window_size * self.window_size, C)

        # Standard multi head attention inside each window
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # [B*num_windows, num_heads, window_area, head_dim]
        q = q.reshape(-1, self.window_size * self.window_size, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(-1, self.window_size * self.window_size, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(-1, self.window_size * self.window_size, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: [B*num_windows, num_heads, window_area, window_area]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        out = self.proj(out)

        # Merge windows back to full feature map
        # [B, Nh, Nw, Wh, Ww, C]
        out = out.reshape(
            B,
            num_windows_h,
            num_windows_w,
            self.window_size,
            self.window_size,
            C,
        )

        out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.reshape(B, Hp, Wp, C)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :H, :W, :]

        # Back to [B, N, C]
        out = out.reshape(B, N, C)
        return out