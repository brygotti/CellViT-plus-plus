# Adapted from: https://github.com/HistAI/hibou/blob/c453bbe4dab0fec6f7df343b09ea87048629c58d/hibou/models/cellvit/cellvit.py

from pathlib import Path
from typing import List, Union, Literal

import torch

from cellvit.models.cell_segmentation.backbones import HibouEncoder
from cellvit.models.cell_segmentation.cellvit import CellViT


class CellViTHibou(CellViT):
    """CellViT with Hibou backbone settings (https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/Checkpoints/vit256_small_dino.pth)

    Skip connections are shared between branches, but each network has a distinct encoder

    Args:
        hibou_path (Union[Path, str]): Path to pretrained Hibou model
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        hibou_structure (Literal["hibou-l"]): Hibou model type
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer in backbone ViT. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connection . Defaults to 0.
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.
    """

    def __init__(
        self,
        hibou_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        hibou_structure: Literal["Hibou-L"],
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        regression_loss: bool = False,
    ):
        if hibou_structure.upper() == "HIBOU-L":
            self.init_hibou_l()
        else:
            raise NotImplementedError("Unknown ViT-Hibou backbone structure")

        self.input_channels = 3  # RGB
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

        self.patch_size = 14
        self.hibou_path = hibou_path
        self.encoder = HibouEncoder(
            path=hibou_path,
            extract_layers=self.extract_layers,
            num_classes=num_tissue_classes,
            dropout_rate=drop_rate,
            attention_dropout_rate=attn_drop_rate,
        )

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        """Forward pass

        Args:
            x (torch.Tensor): Images in BCHW style
            retrieve_tokens (bool, optional): If tokens of ViT should be returned as well. Defaults to False.

        Returns:
            dict: Output for all branches:
                * tissue_types: Raw tissue type prediction. Shape: (B, num_tissue_classes)
                * nuclei_binary_map: Raw binary cell segmentation predictions. Shape: (B, 2, H, W)
                * hv_map: Binary HV Map predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Raw binary nuclei type preditcions. Shape: (B, num_nuclei_classes, H, W)
                * [Optional, if retrieve tokens]: tokens
                * [Optional, if regression loss]:
                * regression_map: Regression map for binary prediction. Shape: (B, 2, H, W)
        """

        # HIBOU-TODO: why ?
        # assert (
        #     x.shape[-2] % self.patch_size == 0
        # ), "Img must have a shape of that is divisible by patch_size (token_size)"
        # assert (
        #     x.shape[-1] % self.patch_size == 0
        # ), "Img must have a shape of that is divisible by patch_size (token_size)"

        out_dict = {}

        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = classifier_logits

        z0, z1, z2, z3, z4 = x, *z

        # z0 = torchvision.transforms.functional.resize(z0, (256,256))

        # performing reshape for the convolutional layers and upsampling (restore spatial dimension)
        # HIBOU-TODO: why ?
        # patch_dim = [int(d / self.patch_size) for d in [x.shape[-2], x.shape[-1]]]
        patch_dim = [16, 16]
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

    # HIBOU-TODO: why those three methods below ?
    def unfreeze_output_layers(self):
        """Unfreeze output layers to train them"""
        extract_layers = [f"blocks.{i - 1}" for i in self.extract_layers]
        # extract_layers += [f'blocks.{i - 2}' for i in self.extract_layers]
        print("*" * 50)
        print(
            "Trainable parameters before unfreezing output layers:",
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
        )
        for layer_name, p in self.encoder.named_parameters():
            for el in extract_layers:
                if el in layer_name:
                    p.requires_grad = True
        print(
            "Trainable parameters after unfreezing output layers:",
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
        )
        print("*" * 50)

    def unfreeze_output_layers_2(self):
        """Unfreeze output layers to train them"""
        extract_layers = [f"blocks.{i - 1}" for i in self.extract_layers]
        extract_layers += [f"blocks.{i - 2}" for i in self.extract_layers]
        print("*" * 50)
        print(
            "Trainable parameters before unfreezing output layers:",
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
        )
        for layer_name, p in self.encoder.named_parameters():
            for el in extract_layers:
                if el in layer_name:
                    p.requires_grad = True
        print(
            "Trainable parameters after unfreezing output layers:",
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
        )
        print("*" * 50)

    def unfreeze_output_layers_3(self):
        """Unfreeze output layers to train them"""
        extract_layers = [f"blocks.{i - 1}" for i in self.extract_layers]
        extract_layers += [f"blocks.{i - 2}" for i in self.extract_layers]
        extract_layers += [f"blocks.{i - 3}" for i in self.extract_layers]
        print("*" * 50)
        print(
            "Trainable parameters before unfreezing output layers:",
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
        )
        for layer_name, p in self.encoder.named_parameters():
            for el in extract_layers:
                if el in layer_name:
                    p.requires_grad = True
        print(
            "Trainable parameters after unfreezing output layers:",
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
        )
        print("*" * 50)

    def init_hibou_l(self):
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.extract_layers = [6, 12, 18, 24]
