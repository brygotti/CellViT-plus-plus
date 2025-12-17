"""Backbone wrapper for mmVIRTUES foundation model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


def _ensure_mmvirtues_on_path(mmvirtues_root: Path) -> None:
    """Add the mmVIRTUES codebase (and helper utils) to ``sys.path`` if missing."""

    root = str(mmvirtues_root.resolve())
    utils_root = str((mmvirtues_root / "datasets_loading").resolve())

    for p in (root, utils_root):
        if p not in sys.path:
            sys.path.insert(0, p)


class MMVirtuesEncoder(nn.Module):
    """Adapter that exposes mmVIRTUES encoder outputs in CellViT-friendly format."""

    def __init__(
        self,
        weights_dir: Path | str,
        extract_layers: Sequence[int] = (6, 12, 18, 24),
        num_classes: int = 0,
        mmvirtues_root: Path | str | None = None,
        resize_to: int = 224,
    ) -> None:
        super().__init__()

        self.weights_dir = Path(weights_dir)
        self.mmvirtues_root = (
            Path(mmvirtues_root)
            if mmvirtues_root is not None
            else self.weights_dir.parent
        )
        self.resize_to = resize_to
        self.extract_layers = list(extract_layers)

        self.marker_embeddings_dir = self._ensure_marker_embeddings()

        _ensure_mmvirtues_on_path(self.mmvirtues_root)
        from omegaconf import OmegaConf
        from mmvirtues_modules import build_mmvirtues_model
        

        cfg = OmegaConf.load(self.weights_dir / "config.yaml")
        # Override the marker embedding path before mmVIRTUES rewrites it
        cfg.marker_embedding_dir = str(self.marker_embeddings_dir)
        self.model = build_mmvirtues_model(
            cfg,
            only_teacher=True,
            ckpt_path=str(self.weights_dir / "teacher_checkpoint.pth"),
        )

        self.embed_dim = getattr(self.model, "embed_dim", 1024)
        self.patch_size = getattr(self.model, "patch_size", 14)
        self.num_register_tokens = getattr(self.model, "num_register_tokens", 0)
        self.num_he_patches = (self.resize_to // self.patch_size) ** 2

        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def _ensure_marker_embeddings(self) -> Path:
        """Resolve marker embeddings directory in a portable way.

        We prefer using paths inside ``mmvirtues_root`` (which is what the example
        notebook/configs provide) and avoid hard-coded locations like ``/workspace``.
        """

        source_dir = self.mmvirtues_root / "esm2_t30_150M_UR50D"
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Marker embeddings not found. Expected {source_dir} (set mmvirtues_root correctly)."
            )

        # Some setups may already have a stable symlink name; if it exists, prefer it.
        symlink_dir = self.mmvirtues_root / "marker_embeddings_symlink"
        if symlink_dir.exists():
            return symlink_dir

        # Create a local symlink for compatibility (best-effort).
        try:
            symlink_dir.symlink_to(source_dir)
            return symlink_dir
        except OSError:
            # If symlinks are not permitted, just return the real directory.
            return source_dir

    def _keep_he_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Keep CLS + the last HE patch tokens.

        mmVIRTUES token ordering can include extra tokens (register, multiplex, proteins).
        For CellViT we only need CLS + HE patch tokens, which are appended at the end.
        """

        if tokens.shape[1] < 1 + self.num_he_patches:
            raise ValueError(
                f"Unexpected token sequence length {tokens.shape[1]} (need >= {1 + self.num_he_patches})."
            )

        he_tokens = tokens[:, -self.num_he_patches :, :]
        return torch.cat((tokens[:, :1, :], he_tokens), dim=1)

    def forward(
        self,
        he_image: torch.Tensor,
        multiplex: Optional[Sequence[Optional[torch.Tensor]]] = None,
        protein_ids: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, None, List[torch.Tensor]]:
        # resize to training resolution expected by mmVIRTUES
        he_resized = TF.resize(he_image, [self.resize_to, self.resize_to])
        batch_size = he_resized.shape[0]

        he_list = [[img] for img in he_resized]
        multiplex_list = (
            multiplex
            if multiplex is not None
            else [[None] for _ in range(batch_size)]
        )
        proteins_list = (
            protein_ids
            if protein_ids is not None
            else [[None] for _ in range(batch_size)]
        )

        tokens_list = [
            self.model.prepare_tokens_with_masks(
                mux, he, prot, technology=None, masks_multiplex=None, masks_he=None
            )
            for mux, he, prot in zip(multiplex_list, he_list, proteins_list)
        ]

        intermediate: List[torch.Tensor] = []
        x_list = tokens_list
        for idx, blk in enumerate(self.model.blocks, start=1):
            x_list = blk(x_list)
            if idx in self.extract_layers:
                cat_tokens = torch.cat(x_list, dim=0)
                intermediate.append(self._keep_he_tokens(cat_tokens))

        cls_tokens = torch.cat([self.model.norm(x)[:, 0, :] for x in x_list], dim=0)
        logits = self.head(cls_tokens)

        return logits, None, intermediate