# -*- coding: utf-8 -*-
"""CellViT variant that uses the mmVIRTUES backbone."""

from pathlib import Path
from typing import Union

import math
import torch

from cellvit.models.cell_segmentation.backbones_mmvirtues import MMVirtuesEncoder
from cellvit.models.cell_segmentation.cellvit import CellViT


class CellViTMMVirtues(CellViT):
    """Cell segmentation model with an mmVIRTUES encoder."""

    def __init__(
        self,
        mmvirtues_weights_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        mmvirtues_root: Union[Path, str, None] = None,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        regression_loss: bool = False,
    ) -> None:
        self.mmvirtues_weights_path = Path(mmvirtues_weights_path)
        self.mmvirtues_root = (
            Path(mmvirtues_root) if mmvirtues_root is not None else self.mmvirtues_weights_path.parent
        )

        encoder = MMVirtuesEncoder(
            weights_dir=self.mmvirtues_weights_path,
            extract_layers=(6, 12, 18, 24),
            num_classes=num_tissue_classes,
            mmvirtues_root=self.mmvirtues_root,
        )

        self.embed_dim = encoder.embed_dim
        self.depth = len(encoder.model.blocks)
        self.num_heads = encoder.model.num_heads
        self.extract_layers = encoder.extract_layers
        self.patch_size = encoder.patch_size
        self.base_res = encoder.resize_to
        self.input_channels = 3
        self.mlp_ratio = 4
        self.qkv_bias = True

        super().__init__(
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            embed_dim=self.embed_dim,
            input_channels=self.input_channels,
            depth=self.depth,
            num_heads=self.num_heads,
            extract_layers=self.extract_layers,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            regression_loss=regression_loss,
        )

        self.encoder = encoder

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        out_dict: dict = {}

        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = classifier_logits

        z0, z1, z2, z3, z4 = x, *z

        # tokens -> feature maps
        # mmVIRTUES operates on a fixed internal resolution; derive the spatial grid
        # from the number of patch tokens rather than from input size / patch size.
        num_patches = int(z4.shape[1] - 1)
        side = int(math.isqrt(num_patches))
        if side * side != num_patches:
            raise ValueError(
                f"Expected a square number of patch tokens, got {num_patches} (seq={int(z4.shape[1])})."
            )
        patch_dim = [side, side]

        z4 = z4[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z3 = z3[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z2 = z2[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z1 = z1[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)

        if self.regression_loss:
            nb_map = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )

        out_dict["hv_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.hv_map_decoder
        )
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder
        )

        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict

    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch_decoder,
    ) -> torch.Tensor:
        """Override to align decoder skips when encoder outputs fixed-resolution tokens."""

        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))

        b0 = self.decoder0(z0)
        h0, w0 = b0.shape[-2:]
        h1, w1 = b1.shape[-2:]
        base = self.base_res
        if h0 <= 0 or w0 <= 0:
            h0, w0 = max(h1, 1, base), max(w1, 1, base)
            b0 = torch.zeros((b1.shape[0], b0.shape[1], h0, w0), device=b1.device, dtype=b1.dtype)
        if h1 <= 0 or w1 <= 0:
            h1, w1 = max(h0, 1, base), max(w0, 1, base)
            b1 = torch.zeros((b0.shape[0], b1.shape[1], h1, w1), device=b0.device, dtype=b0.dtype)
        target_h = max(h0, h1, base, 1)
        target_w = max(w0, w1, base, 1)

        if b0.shape[-2:] != (target_h, target_w):
            b0 = torch.nn.functional.interpolate(b0, size=(target_h, target_w), mode="bilinear", align_corners=False)
        if b1.shape[-2:] != (target_h, target_w):
            b1 = torch.nn.functional.interpolate(b1, size=(target_h, target_w), mode="bilinear", align_corners=False)

        return branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))