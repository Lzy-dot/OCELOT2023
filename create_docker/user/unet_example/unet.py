
import os
import numpy as np
from skimage import feature
import cv2
from util.constants import SAMPLE_SHAPE
from monai.transforms import ScaleIntensityd
import torch
import torch.nn as nn
import torch.nn.functional as F1
import torchvision.transforms.functional as F1_tv
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from re import T
from typing import Sequence, Tuple, Type, Union
import torch
import torch.nn as nn
import cv2
from torchvision import models
import torchvision.transforms.functional as F2
import json
from monai.utils import ensure_tuple_rep, optional_import
"""
resnet backbone
"""

import math
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, List, Tuple, Union, Type, Optional
import torch.multiprocessing

from collections import OrderedDict
from functools import partial
from pathlib import Path


import math
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, List, Tuple, Union

import torch
import torch.multiprocessing
import torch.nn as nn
from torchvision import transforms

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


from skimage import exposure


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)

class Conv2DBlock(nn.Module):
    """Conv2DBlock with convolution followed by batch-normalisation, ReLU activation and dropout

    Args:
        in_channels (int): Number of input channels for convolution
        out_channels (int): Number of output channels for convolution
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
                groups=in_channels,  # 设置 groups 参数为输入通道数，实现深度可分离卷积
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,  # 逐点卷积的 kernel size 为 1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """Deconvolution block with ConvTranspose2d followed by Conv2d, batch-normalisation, ReLU activation and dropout

    Args:
        in_channels (int): Number of input channels for deconv block
        out_channels (int): Number of output channels for deconv and convolution.
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=1, 
                padding=(kernel_size - 1) // 2, 
                groups=out_channels
            ),
            nn.Conv2d(in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))



class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        scale: float = 0.5,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.scale = scale
        # self.Depth_Adapter = Adapter(dim, skip_connect=False)  # no skip connection

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )

        self.window_size = window_size

    def forward(self, x: torch.Tensor, mode_tc) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x, mode_tc)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        xn = self.norm2(x)
        x = x + self.mlp(xn)

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor, mode_tc) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x, mode_tc).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)

        return x


def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F1.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F1.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x




class CellViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        input_channels: int,
        drop_rate: float = 0,
    ):
        # For simplicity, we will assume that extract layers must have a length of 4
        super().__init__()
        self.patch_size = 16
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.input_channels = input_channels
        self.drop_rate = drop_rate
        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        # version with shared skip_connections
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )  # skip connection after positional encoding, shape should be H, W, 64
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )  # skip connection 1
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )  # skip connection 2
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )  # skip connection 3

        # self.nuclei_binary_map_decoder = self.create_upsampling_branch(2)
        # self.hv_map_decoder = self.create_upsampling_branch(2)
        self.decoder = self.create_upsampling_branch(
            self.num_classes
        )


 

    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch_decoder: nn.Sequential,
    ) -> torch.Tensor:
        """Forward upsample branch

        Args:
            z0 (torch.Tensor): Highest skip
            z1 (torch.Tensor): 1. Skip
            z2 (torch.Tensor): 2. Skip
            z3 (torch.Tensor): 3. Skip
            z4 (torch.Tensor): Bottleneck
            branch_decoder (nn.Sequential): Branch decoder network

        Returns:
            torch.Tensor: Branch Output
        """
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2) # 这里得到的b2刚好是下采样4倍后的特征 [H/4 W/4 256],可以在这里加入tissue的预测结果,这里的decoder3_upsampler的输出通道由256变为258，decoder2_upsampler的输入通道由256*2变为256*2+2
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))

        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

        return branch_output

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        """Create Upsampling branch

        Args:
            num_classes (int): Number of output classes

        Returns:
            nn.Module: Upsampling path
        """
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(
                self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )

        return decoder


class ViTCellViTDeit(ImageEncoderViT):
    def __init__(
        self,
        extract_layers: List[int],
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            out_chans,
            qkv_bias,
            norm_layer,
            act_layer,
            use_abs_pos,
            use_rel_pos,
            rel_pos_zero_init,
            window_size,
            global_attn_indexes,
        )
        self.extract_layers = extract_layers

    def forward(self, x: torch.Tensor, mode_tc) -> torch.Tensor:
        extracted_layers = []
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            token_size = x.shape[1]
            x = x + self.pos_embed[:, :token_size, :token_size, :]

        for depth, blk in enumerate(self.blocks):
            x = blk(x, mode_tc)
            if depth + 1 in self.extract_layers:
                extracted_layers.append(x)

        return extracted_layers


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_t_q: nn.Module,
            linear_a_c_q: nn.Module,
            linear_b_t_q: nn.Module,
            linear_b_c_q: nn.Module,
            linear_a_t_v: nn.Module,
            linear_a_c_v: nn.Module,
            linear_b_t_v: nn.Module,
            linear_b_c_v: nn.Module,
    ):
        super().__init__()
        # self.mode = mode
        self.qkv = qkv
        self.linear_a_t_q = linear_a_t_q
        self.linear_a_c_q = linear_a_c_q
        self.linear_b_t_q = linear_b_t_q
        self.linear_b_c_q = linear_b_c_q
        self.linear_a_t_v = linear_a_t_v
        self.linear_a_c_v = linear_a_c_v
        self.linear_b_t_v = linear_b_t_v
        self.linear_b_c_v = linear_b_c_v
        self.dim = qkv.in_features
        # self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x, mode_tc):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        if mode_tc == 'T':
            new_q = self.linear_b_t_q(self.linear_a_t_q(x))
            new_v = self.linear_b_t_v(self.linear_a_t_v(x))
            qkv[:, :, :, : self.dim] += new_q
            qkv[:, :, :, -self.dim:] += new_v
        else:
            new_q = self.linear_b_c_q(self.linear_a_c_q(x))
            new_v = self.linear_b_c_v(self.linear_a_c_v(x))
            qkv[:, :, :, : self.dim] += new_q
            qkv[:, :, :, -self.dim:] += new_v       
        return qkv


class CellViTSAM(nn.Module):
    """CellViT with SAM backbone settings

    Skip connections are shared between branches

    Args:
        model_path (Union[Path, str]): Path to pretrained SAM model
        num_cell_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        vit_structure (Literal["SAM-B", "SAM-L", "SAM-H"]): SAM model type
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.

    Raises:
        NotImplementedError: Unknown SAM configuration
    """

    def __init__(
        self,
        num_cell_classes: int,
        num_tissue_classes: int,
        vit_structure,
        drop_rate: float = 0,
        freeze_encoder : bool = True
    ):
        if vit_structure == "SAM-B":
            self.init_vit_b()
        elif vit_structure == "SAM-L":
            self.init_vit_l()
        elif vit_structure == "SAM-H":
            self.init_vit_h()
        else:
            raise NotImplementedError("Unknown ViT-SAM backbone structure")
        self.patch_size = 16
        self.input_channels = 3  # RGB
        self.mlp_ratio = 4
        self.qkv_bias = True
        # self.num_cell_classes = num_cell_classes
        super().__init__()
        self.prompt_embed_dim = 256
        self.encoder = ViTCellViTDeit(
            extract_layers=self.extract_layers,
            depth=self.depth,
            embed_dim=self.embed_dim,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=self.num_heads,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=self.encoder_global_attn_indexes,
            window_size=14,
            out_chans=self.prompt_embed_dim,
        )

        self.decoder_t = CellViT(num_classes=num_tissue_classes,embed_dim=self.embed_dim,input_channels=self.input_channels,drop_rate=0.1)
        self.decoder_c = CellViT(num_classes=num_cell_classes,embed_dim=self.embed_dim,input_channels=self.input_channels,drop_rate=0.3)
        
        # create for storage, then we can init them or load weights
        self.w_As_t = []  # These are linear layers
        self.w_Bs_t = []
        self.w_As_c = []  # These are linear layers
        self.w_Bs_c = []
        # Here, we do the surgery
        for t_layer_i, blk in enumerate(self.encoder.blocks):
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q_t = nn.Linear(self.dim, 6, bias=False)
            w_b_linear_q_t = nn.Linear(6, self.dim, bias=False)
            w_a_linear_v_t = nn.Linear(self.dim, 6, bias=False)
            w_b_linear_v_t = nn.Linear(6, self.dim, bias=False)

            w_a_linear_q_c = nn.Linear(self.dim, 6, bias=False)
            w_b_linear_q_c = nn.Linear(6, self.dim, bias=False)
            w_a_linear_v_c = nn.Linear(self.dim, 6, bias=False)
            w_b_linear_v_c = nn.Linear(6, self.dim, bias=False)
            blk.attn.qkv = _LoRA_qkv(
                    w_qkv_linear,
                    w_a_linear_q_t,
                    w_a_linear_q_c,
                    # w_a_linear_q,
                    w_b_linear_q_t,
                    w_b_linear_q_c,
                    # w_b_linear_q,
                    w_a_linear_v_t,
                    w_a_linear_v_c,
                    # w_a_linear_v,
                    w_b_linear_v_t,
                    w_b_linear_v_c,
                    # w_b_linear_v,
                )
            self.w_As_t.append(w_a_linear_q_t)
            self.w_Bs_t.append(w_b_linear_q_t)
            self.w_As_t.append(w_a_linear_v_t)
            self.w_Bs_t.append(w_b_linear_v_t)
            self.w_As_c.append(w_a_linear_q_c)
            self.w_Bs_c.append(w_b_linear_q_c)
            self.w_As_c.append(w_a_linear_v_c)
            self.w_Bs_c.append(w_b_linear_v_c)
        # self.reset_parameters()

        # if freeze_encoder:
            # self.freeze_encoder()

    def reset_parameters(self) -> None:
        for w_A in self.w_As_t:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs_t:
            nn.init.zeros_(w_B.weight)

        for w_A in self.w_As_c:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs_c:
            nn.init.zeros_(w_B.weight)
            
    def load_pretrained_encoder(self, model_path):
        """Load pretrained SAM encoder from provided path

        Args:
            model_path (str): Path to SAM model
        """
        state_dict = torch.load(str(model_path), map_location="cpu")
        image_encoder = self.encoder
        msg = image_encoder.load_state_dict(state_dict, strict=False)
        print(f"Loading checkpoint: {msg}")
        self.encoder = image_encoder

    def forward(self, x: torch.Tensor, x_t, mode_tc):
        assert (
            x.shape[-2] % self.patch_size == 0
        ), "Img must have a shape of that is divisble by patch_soze (token_size)"
        assert (
            x.shape[-1] % self.patch_size == 0
        ), "Img must have a shape of that is divisble by patch_soze (token_size)"
        z = self.encoder(x,mode_tc)
        z0, z1, z2, z3, z4 = x, *z
        
        # performing reshape for the convolutional layers and upsampling (restore spatial dimension)
        # 在Patch_embed时变换了通道顺序(B C H W)->(B H W C)，之后需要变回来
        z4 = z4.permute(0, 3, 1, 2)
        z3 = z3.permute(0, 3, 1, 2)
        z2 = z2.permute(0, 3, 1, 2)
        z1 = z1.permute(0, 3, 1, 2)
        if mode_tc == 'T':
            out = self.decoder_t._forward_upsample(
                z0, z1, z2, z3, z4, self.decoder_t.decoder
            )
        else:
            out = self.decoder_c._forward_upsample(
                z0+x_t[0], z1+x_t[1], z2+x_t[1], z3+x_t[1], z4+x_t[1], self.decoder_c.decoder
            )
        return out

    def init_vit_b(self):
        self.embed_dim = 768
        self.depth = 12
        self.num_heads = 12
        self.encoder_global_attn_indexes = [2, 5, 8, 11]
        self.extract_layers = [3, 6, 9, 12]

    def init_vit_l(self):
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.encoder_global_attn_indexes = [5, 11, 17, 23]
        self.extract_layers = [6, 12, 18, 24]

    def init_vit_h(self):
        self.embed_dim = 1280
        self.depth = 32
        self.num_heads = 16
        self.encoder_global_attn_indexes = [7, 15, 23, 31]
        self.extract_layers = [8, 16, 24, 32]

class SAM_Model(nn.Module):
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        feature_size: int = 64,
        feature_size2: int = 24,
        spatial_dims: int = 2,
        ):
        super(SAM_Model, self).__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.feature_size2 = feature_size2
        self.spatial_dims = spatial_dims
        self.branch = CellViTSAM(num_cell_classes=3,num_tissue_classes=2,vit_structure='SAM-H',freeze_encoder=True)
        self.cell_conv0 = nn.Conv2d(2,3,kernel_size=1,stride=1,padding=0)
        self.cell_conv1 = nn.Conv2d(2,1280,kernel_size=1,stride=1,padding=0)
    def forward(self, id, tissue, cell, mask_meta,cropped_tissue=None,mode=None):
        
        
        if mode == 't':
       
            pred_tissue0 = self.branch(tissue, None,'T') # [B C H W] [B 2 512 512]
            pred_tissue = pred_tissue0[:,0:2,:,:]
            res0 = []
            # 直接加到输入上
            # print("H1,W1",H1,W1)
            id = [id]
            for i in range(len(id)):
                tissue_pred = pred_tissue[i]
                mask_meta_tmp = mask_meta[i]
                # 根据mask创建裁剪框
                bbox = torch.where(mask_meta_tmp.squeeze(0) == 1)
                y_min = bbox[0].min().item()
                y_max = bbox[0].max().item()
                x_min = bbox[1].min().item()
                x_max = bbox[1].max().item()
                # 裁剪预测图
                cropped_tissue = tissue_pred[:, y_min:y_max+1, x_min:x_max+1]
                res0.append(cropped_tissue.unsqueeze(0))
            return torch.cat(res0, dim=0)
        else:
            _,C1,H1,W1 = cell.shape
            re_size = [int(H1), int(W1)]
            re_size1 = [int(H1/16), int(W1/16)]
            cropped_tissue0 = self.cell_conv0(cropped_tissue)
            cropped_tissue1 = self.cell_conv1(cropped_tissue)
            resized_pred0 = F2.resize(cropped_tissue0, re_size)
            resized_pred1 = F2.resize(cropped_tissue1, re_size1)
            pred_cell = self.branch(cell,[resized_pred0,resized_pred1],'C')
            return pred_cell
        return pred_tissue0, pred_cell

class PytorchUnetCellModel():
    """
    U-NET model for cell detection implemented with the Pytorch library

    NOTE: this model does not utilize the tissue patch but rather
    only the cell patch.

    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """
    def __init__(self, metadata):
        self.device = torch.device('cuda:0')
        self.metadata = metadata
        self.resize_to = (512, 512) # The model is trained with 1024 resolution
        # RGB images and 2 class prediction
        self.n_classes =  2 # Two cell classes and background
        # self.unet = CellViTSAM(num_cell_classes=3,num_tissue_classes=2,vit_structure='SAM-H',freeze_encoder=True)
        self.unet = SAM_Model(img_size=self.resize_to, in_channels=3, out_channels=self.n_classes)
        self.unet1 = SAM_Model(img_size=self.resize_to, in_channels=3, out_channels=self.n_classes)
        self.unet2 = SAM_Model(img_size=self.resize_to, in_channels=3, out_channels=self.n_classes)
        self.unet3 = SAM_Model(img_size=self.resize_to, in_channels=3, out_channels=self.n_classes)
        # self.unet = UNet(n_channels=3, n_classes=self.n_classes)
      
        
        self.load_checkpoint()
        self.unet = self.unet.to(self.device)
        self.unet.eval()
        self.unet1 = self.unet1.to(self.device)
        self.unet1.eval()
        self.unet2 = self.unet2.to(self.device)
        self.unet2.eval()
        self.unet3 = self.unet3.to(self.device)
        self.unet3.eval()

    def load_checkpoint(self):
        """Loading the trained weights to be used for validation"""
        _curr_path = os.path.split(__file__)[0]
        _path_to_checkpoint = os.path.join(_curr_path, "checkpoints/seg_celltissue512_lrsame_enc2_f1_final_model.pth")
        state_dict = torch.load(_path_to_checkpoint, map_location=torch.device('cpu'))['model_state_dict']
        self.unet.load_state_dict(state_dict, strict=False)
        print("Weights were successfully loaded!")
        _curr_path = os.path.split(__file__)[0]
        _path_to_checkpoint = os.path.join(_curr_path, "checkpoints/seg_celltissue512_lrsame_enc2_f1_final_model_2.pth")
        state_dict = torch.load(_path_to_checkpoint, map_location=torch.device('cpu'))['model_state_dict']
        self.unet1.load_state_dict(state_dict, strict=False)
        print("Weights were successfully loaded!")

        _curr_path = os.path.split(__file__)[0]
        _path_to_checkpoint = os.path.join(_curr_path, "checkpoints/seg_celltissue512_lrsame_enc2_f1_final_model_3.pth")
        state_dict = torch.load(_path_to_checkpoint, map_location=torch.device('cpu'))['model_state_dict']
        self.unet2.load_state_dict(state_dict, strict=False)
        print("Weights were successfully loaded!")

        _curr_path = os.path.split(__file__)[0]
        _path_to_checkpoint = os.path.join(_curr_path, "checkpoints/seg_celltissue512_lrsame_enc2_f1_final_model_4.pth")
        state_dict = torch.load(_path_to_checkpoint, map_location=torch.device('cpu'))['model_state_dict']
        self.unet3.load_state_dict(state_dict, strict=False)
        print("Weights were successfully loaded!")

    def prepare_input(self, cell_patch, tissue_patch, mask_meta):
        """This function prepares the cell patch array to be forwarded by
        the model
        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        Returns
        -------
            torch.tensor of shape [1, 3, 1024, 1024] where the first axis is the batch
            dimension
        """
        pre_img_data = np.zeros(cell_patch.shape, dtype=np.uint8)
        for i in range(3):
            img_channel_i = cell_patch[:,:,i]
            if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
        test_npy01 = pre_img_data/np.max(pre_img_data)
        cell_patch = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(self.device)
        
        pre_img_data = np.zeros(tissue_patch.shape, dtype=np.uint8)
        for i in range(3):
            img_channel_i = tissue_patch[:,:,i]
            if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
        test_npy01 = pre_img_data/np.max(pre_img_data)
        tissue_patch = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(self.device)
        
        mask_meta = torch.from_numpy(mask_meta).unsqueeze(2).permute(2, 0, 1).unsqueeze(0)
        
        if self.resize_to is not None:
            cell_patch= F1.interpolate(
                    cell_patch, size=self.resize_to, mode="area"
            ).detach()
            tissue_patch= F1.interpolate(
                tissue_patch, size=self.resize_to, mode="area"
            ).detach()
            mask_meta= F1.interpolate(
                mask_meta, size=self.resize_to, mode="nearest"
            ).detach()
        return cell_patch, tissue_patch, mask_meta
        
    def find_cells(self, heatmap):
        """This function detects the cells in the output heatmap

        Parameters
        ----------
        heatmap: torch.tensor
            output heatmap of the model,  shape: [1, 3, 1024, 1024]

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        arr = heatmap[0,:,:,:].cpu().detach().numpy()
        # arr = np.transpose(arr, (1, 2, 0)) # CHW -> HWC

        bg, pred_wo_bg = np.split(arr, (1,), axis=0) # Background and non-background channels
        bg = np.squeeze(bg, axis=0)
        obj = 1.0 - bg

        arr = cv2.GaussianBlur(obj, (0, 0), sigmaX=3)
        peaks = feature.peak_local_max(
            arr, min_distance=7, exclude_border=0, threshold_abs=0.0
        ) # List[y, x]

        maxval = np.max(pred_wo_bg, axis=0)
        # pred_wo_bg[0]=cv2.blur(pred_wo_bg[0], (15, 15))
        # pred_wo_bg[1]=cv2.blur(pred_wo_bg[1], (15, 15))
        maxcls_0 = np.argmax(pred_wo_bg, axis=0)

        # Filter out peaks if background score dominates
        peaks = np.array([peak for peak in peaks if bg[peak[0], peak[1]] < maxval[peak[0], peak[1]]])
        if len(peaks) == 0:
            return []

        # Get score and class of the peaks
        scores = maxval[peaks[:, 0], peaks[:, 1]]
        peak_class = maxcls_0[peaks[:, 0], peaks[:, 1]]

        predicted_cells = [(x, y, c + 1, float(s)) for [y, x], c, s in zip(peaks, peak_class, scores)]

        return predicted_cells

    def post_process(self, logits):
        """This function applies some post processing to the
        output logits
        
        Parameters
        ----------
        logits: torch.tensor
            Outputs of U-Net

        Returns
        -------
            torch.tensor after post processing the logits
        """
        if self.resize_to is not None:
            logits = F1.interpolate(logits, size=SAMPLE_SHAPE[:2],
                mode='bilinear', align_corners=False
            )
        return torch.softmax(logits, dim=1)

    def __call__(self, cell_patch, tissue_patch, pair_id):
        """This function detects the cells in the cell patch using Pytorch U-Net.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8] 
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        crop_size=256
        mask = np.zeros((1024, 1024))
        offset_x = self.metadata[pair_id]['patch_x_offset'] * 1024
        offset_y = self.metadata[pair_id]['patch_y_offset'] * 1024
        top_left = (int(offset_x - crop_size // 2), int(offset_y - crop_size // 2))
        mask[top_left[1]:top_left[1]+crop_size, top_left[0]:top_left[0]+crop_size]=1
        mask_meta = mask
        cell_patch,tissue_patch,mask_meta = self.prepare_input(cell_patch,tissue_patch,mask_meta)
        with torch.no_grad():
            # for i in range(4):
            tissue = self.unet(pair_id,tissue_patch,cell_patch,mask_meta,mode='t')
            tissue += self.unet1(pair_id,tissue_patch,cell_patch,mask_meta,mode='t')
            tissue += self.unet2(pair_id,tissue_patch,cell_patch,mask_meta,mode='t')
            tissue += self.unet3(pair_id,tissue_patch,cell_patch,mask_meta,mode='t')
            tissue = tissue/4
            logit = self.unet(pair_id,tissue_patch,cell_patch,mask_meta,cropped_tissue=tissue,mode='c')
            logit += self.unet1(pair_id,tissue_patch,cell_patch,mask_meta,cropped_tissue=tissue,mode='c')
            logit += self.unet2(pair_id,tissue_patch,cell_patch,mask_meta,cropped_tissue=tissue,mode='c')
            logit += self.unet3(pair_id,tissue_patch,cell_patch,mask_meta,cropped_tissue=tissue,mode='c')
         
        heatmap = self.post_process(logit/4)
        return self.find_cells(heatmap)
