# -*- coding: utf-8 -*-
# Spatial Proteomics Dataset for CellViT with MM-VirTues
#
# Loads paired Spatial Proteomics (multiplex immunofluorescence) and H&E images
# with cell segmentation annotations

import sys
from pathlib import Path
from typing import Callable, Optional, Tuple
import logging

import torch
import numpy as np
import pandas as pd
from PIL import Image

from omegaconf import OmegaConf

from cellvit.models.cell_segmentation.backbones_mmvirtues import _ensure_mmvirtues_on_path


def _import_mmvirtues_dataset_modules(mmvirtues_root: Path):
    """Import mmVIRTUES dataset utilities after ensuring the path.

    This avoids hard-coded absolute paths (e.g. /mnt/...) and makes the dataset portable.
    """

    _ensure_mmvirtues_on_path(mmvirtues_root)
    try:
        from datasets_loading.datasets.mm_base import build_mm_datasets
        from datasets_loading.utils.marker_utils import load_marker_embeddings
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "Could not import mmVIRTUES 'datasets_loading' modules. "
            "Set dataset_path/mmvirtues_root to the mmvirtues_orion_dataset 'virtues_example' folder "
            "that contains a 'datasets_loading' directory."
        ) from exc
    return build_mm_datasets, load_marker_embeddings

logger = logging.getLogger(__name__)

class SpatialProteomicsDataset(torch.utils.data.Dataset):
    """Dataset for paired Spatial Proteomics + H&E images with cell segmentation"""
    
    def __init__(
        self,
        dataset_path: str = "/mnt/course-cs-433-group08/scratch/mmvirtues_orion_dataset/virtues_example",
        orion_subset_path: str = "orion_subset",
        protein_embeddings_path: str = "esm2_t30_150M_UR50D",
        crop_size: int = 224,
        use_tissue_masks: bool = True,
        use_cell_type_masks: bool = True,
        transforms: Optional[Callable] = None,
        mmvirtues_root: Optional[str] = None,
    ):
        """
        Args:
            dataset_path: Path to MM-VirTues dataset root
            orion_subset_path: Relative path to ORION dataset subset
            protein_embeddings_path: Relative path to ESM2 protein embeddings
            crop_size: Image crop size (MM-VirTues uses 224)
            use_tissue_masks: Whether to load tissue segmentation masks
            use_cell_type_masks: Whether to load cell type annotations
            transforms: Optional data augmentation transforms
        """
        super().__init__()
        
        self.dataset_path = Path(dataset_path)
        self.mmvirtues_root = Path(mmvirtues_root) if mmvirtues_root is not None else self.dataset_path
        self.crop_size = crop_size
        self.transforms = transforms
        self.use_tissue_masks = use_tissue_masks
        self.use_cell_type_masks = use_cell_type_masks
        
        # Load MM-VirTues dataset configuration
        logger.info(f"Loading dataset from {dataset_path}")
        base_cfg = OmegaConf.load(self.dataset_path / "datasets_loading/configs/base_config.yaml")
        base_cfg.marker_embedding_dir = str(self.dataset_path / protein_embeddings_path)
        
        orion_cfg = OmegaConf.load(self.dataset_path / "datasets_loading/configs/orion_subset.yaml")
        self.ds_cfg = OmegaConf.merge(base_cfg, orion_cfg)
        
        # Build dataset (mmVIRTUES external dependency)
        build_mm_datasets, _ = _import_mmvirtues_dataset_modules(self.mmvirtues_root)
        self.mm_dataset = build_mm_datasets(self.ds_cfg)
        
        # Get tissue IDs (sample identifiers)
        self.tissue_ids = self.mm_dataset[0].unimodal_datasets["cycif"].get_tissue_ids()
        logger.info(f"Loaded {len(self.tissue_ids)} tissue samples")
        
        # Load protein ID to name mapping
        self.uniprot_to_name = self.mm_dataset[0].unimodal_datasets["cycif"].get_marker_embedding_index_to_name_dict()
        
        # Paths to annotations
        self.orion_subset_path = self.dataset_path / orion_subset_path
        self.cell_masks_path = self.orion_subset_path / "cell_masks"
        self.tissue_masks_path = self.orion_subset_path / "tissue_masks"
        
        # Load annotations if available
        self.has_cell_masks = (self.cell_masks_path / "segmentation").exists()
        self.has_tissue_masks = self.tissue_masks_path.exists()
        
        if not self.has_cell_masks:
            logger.warning(f"Cell masks not found at {self.cell_masks_path / 'segmentation'}")
        if not self.has_tissue_masks:
            logger.warning(f"Tissue masks not found at {self.tissue_masks_path}")
        
    def __len__(self) -> int:
        return len(self.tissue_ids)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
            - 'sp_image': torch.Tensor [C_sp, crop_size, crop_size]
            - 'he_image': torch.Tensor [3, crop_size, crop_size]
            - 'protein_ids': torch.Tensor [C_sp]
            - 'tissue_id': str
            - 'nuclei_binary_map': Optional[torch.Tensor] [2, H, W]
            - 'nuclei_type_map': Optional[torch.Tensor] [num_classes, H, W]
            - 'hv_map': Optional[torch.Tensor] [2, H, W]
        """
        tissue_id = self.tissue_ids[idx]
        
        # Load SP image (multiplex immunofluorescence)
        sp_image = self.mm_dataset[0].unimodal_datasets["cycif"].get_tissue(tissue_id)
        sp_image = torch.tensor(np.array(sp_image), dtype=torch.float32)
        
        # Load H&E image
        he_image = self.mm_dataset[0].unimodal_datasets["he"].get_tissue(tissue_id)
        he_image = torch.tensor(np.array(he_image), dtype=torch.float32)
        
        # Get protein IDs (marker indices)
        protein_ids = self.mm_dataset[0].unimodal_datasets["cycif"].get_marker_embedding_indices(tissue_id)
        protein_ids = torch.tensor(protein_ids, dtype=torch.long)
        
        # Crop to required size
        if sp_image.shape[-2] > self.crop_size or sp_image.shape[-1] > self.crop_size:
            sp_image = sp_image[:, :self.crop_size, :self.crop_size]
            he_image = he_image[:, :self.crop_size, :self.crop_size]
        
        # Prepare output dict
        output = {
            'sp_image': sp_image,
            'he_image': he_image,
            'protein_ids': protein_ids,
            'tissue_id': tissue_id,
        }
        
        # Load cell segmentation masks if available
        if self.has_cell_masks and self.use_cell_type_masks:
            cell_mask = self._load_cell_mask(tissue_id)
            if cell_mask is not None:
                output['nuclei_binary_map'] = cell_mask['binary']
                output['nuclei_type_map'] = cell_mask['type']
                output['hv_map'] = cell_mask['hv']
        
        # Load tissue segmentation if available
        if self.has_tissue_masks and self.use_tissue_masks:
            tissue_mask = self._load_tissue_mask(tissue_id)
            if tissue_mask is not None:
                output['tissue_mask'] = tissue_mask
        
        # Apply transforms if provided
        if self.transforms:
            output = self.transforms(output)
        
        return output
    
    def _load_cell_mask(self, tissue_id: str) -> Optional[dict]:
        """Load cell segmentation masks for a tissue
        
        Returns dict with:
            - binary: Binary nuclei segmentation [2, H, W]
            - type: Cell type segmentation [num_classes, H, W]
            - hv: Horizontal-Vertical map for instance separation [2, H, W]
        """
        mask_dir = self.cell_masks_path / "segmentation" / tissue_id
        
        if not mask_dir.exists():
            return None
        
        # Look for mask files (format depends on your dataset)
        # This is a placeholder - adjust based on your actual file format
        try:
            # Example: Load instance segmentation mask
            instance_mask_path = mask_dir / "instances.png"
            if instance_mask_path.exists():
                instance_mask = np.array(Image.open(instance_mask_path))
                
                # Convert instance mask to binary mask
                binary_mask = (instance_mask > 0).astype(np.uint8)
                binary_map = torch.zeros(2, *binary_mask.shape, dtype=torch.float32)
                binary_map[0] = torch.from_numpy(1 - binary_mask)  # Background
                binary_map[1] = torch.from_numpy(binary_mask)      # Foreground
                
                # Generate HV map from instance mask (placeholder - needs proper implementation)
                hv_map = self._generate_hv_map(instance_mask)
                
                # Load cell type mask if available
                type_mask_path = mask_dir / "types.png"
                if type_mask_path.exists():
                    type_mask = np.array(Image.open(type_mask_path))
                    num_classes = len(np.unique(type_mask))
                    type_map = torch.zeros(num_classes, *type_mask.shape, dtype=torch.float32)
                    for i in range(num_classes):
                        type_map[i] = torch.from_numpy((type_mask == i).astype(np.float32))
                else:
                    type_map = binary_map  # Use binary if no type info
                
                return {
                    'binary': binary_map[:, :self.crop_size, :self.crop_size],
                    'type': type_map[:, :self.crop_size, :self.crop_size],
                    'hv': hv_map[:, :self.crop_size, :self.crop_size],
                }
        except Exception as e:
            logger.warning(f"Failed to load cell mask for {tissue_id}: {e}")
            return None
    
    def _load_tissue_mask(self, tissue_id: str) -> Optional[torch.Tensor]:
        """Load tissue segmentation mask"""
        mask_path = self.tissue_masks_path / f"{tissue_id}.png"
        
        if not mask_path.exists():
            return None
        
        try:
            mask = np.array(Image.open(mask_path))
            mask = torch.from_numpy(mask).float()
            return mask[:self.crop_size, :self.crop_size]
        except Exception as e:
            logger.warning(f"Failed to load tissue mask for {tissue_id}: {e}")
            return None
    
    def _generate_hv_map(self, instance_mask: np.ndarray) -> torch.Tensor:
        """Generate Horizontal-Vertical map from instance segmentation
        
        This is a placeholder - proper HV map generation needs:
        1. Compute gradient in x and y directions
        2. Normalize by distance to cell center
        
        See: https://github.com/vqdang/hover_net for reference
        """
        # Placeholder: return zeros
        hv_map = torch.zeros(2, *instance_mask.shape, dtype=torch.float32)
        
        # TODO: Implement proper HV map generation
        # from scipy.ndimage import distance_transform_edt
        # For each instance, compute centroid and gradient
        
        return hv_map
    
    def get_protein_names(self, idx: int) -> list:
        """Get human-readable protein names for a sample"""
        tissue_id = self.tissue_ids[idx]
        protein_ids = self.mm_dataset[0].unimodal_datasets["cycif"].get_marker_embedding_indices(tissue_id)
        return [self.uniprot_to_name[pid] for pid in protein_ids]
    
    def load_cell_count(self):
        """Load cell count statistics (required by CellDataset interface)"""
        # TODO: Implement if you have cell count CSV
        pass
    
    def get_sampling_weights_tissue(self, gamma: float = 1) -> torch.Tensor:
        """Get sampling weights by tissue type"""
        # TODO: Implement if you want weighted sampling
        return torch.ones(len(self))
    
    def get_sampling_weights_cell(self, gamma: float = 1) -> torch.Tensor:
        """Get sampling weights by cell type"""
        # TODO: Implement if you want weighted sampling
        return torch.ones(len(self))