# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer_virtues_tokenizer as vits
from . import vision_transformer_virtues_tokenizer as vits_virtues_tokenizer

from .flex_dual_mmvirtues import build_flex_dual_virtues_encoder
from utils.marker_utils import load_marker_embeddings
import copy
from torch import nn
import torch


logger = logging.getLogger("dinov2")

def build_mmvirtues_model(cfg, only_teacher=False, ckpt_path=None):
    cfg.marker_embedding_dir = "/workspace/mmvirtues_orion_dataset/virtues_example/esm2_t30_150M_UR50D" if "esm2_t30_150M_UR50D" in cfg.marker_embedding_dir else cfg.marker_embedding_dir
    esm_embds = load_marker_embeddings(cfg.marker_embedding_dir)
    esm_embds = esm_embds.to(dtype=torch.float16)
    virtues_encoder = build_flex_dual_virtues_encoder(cfg, esm_embds, only_teacher=only_teacher, pos_emb="rope")
    model, _ = build_model_from_cfg(cfg, only_teacher=only_teacher, mx_embed_layer=virtues_encoder, norm_after_he_tokenizer=cfg.model.norm_after_he_tokenizer)

    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        state_dict = state_dict['teacher']
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        print(model.load_state_dict(state_dict, strict=False))
    return model

def build_model_from_cfg(cfg, only_teacher=False, mx_embed_layer=None, norm_after_he_tokenizer=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size, mx_embed_layer=mx_embed_layer, norm_after_he_tokenizer=norm_after_he_tokenizer)

def build_model(args, only_teacher=False, img_size=224, mx_embed_layer=None, norm_after_he_tokenizer=False):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
            mx_embed_layer=mx_embed_layer,
            he_embed_layer=None,
        )


        if mx_embed_layer is None:
            teacher = vits.__dict__[args.arch](**vit_kwargs)
            student = vits.__dict__[args.arch](
                **vit_kwargs,
                drop_path_rate=args.drop_path_rate,
                drop_path_uniform=args.drop_path_uniform,
            )
        else:
            model_vit_size = args.arch
            match model_vit_size:
                case "vit_base": model_vit_dim = 768
                case "vit_large": model_vit_dim = 1024
                case "vit_huge": model_vit_dim = 1280
                case "vit_giant": model_vit_dim = 1408

            assert model_vit_dim == mx_embed_layer.model_dim, f"Model dimension mismatch: {model_vit_dim} != {mx_embed_layer.model_dim}"
            # if mx_embed_layer.model_dim != model_vit_dim:
            #     logger.info("Model dimension mismatch of patch embedding and vit, enabling NormLayer + adding Linear Layer")
            #     mx_embed_layer.norm_after_encoder_decoder = True
            #     mx_embed_layer.encoder.add_module("linear_projection", LinearProjection(mx_embed_layer.model_dim, model_vit_dim))

            teacher = vits_virtues_tokenizer.__dict__[args.arch](**vit_kwargs)
            teacher.mx_embed_layer = copy.deepcopy(mx_embed_layer)

            student = vits_virtues_tokenizer.__dict__[args.arch](
                **vit_kwargs,
                drop_path_rate=args.drop_path_rate,
                drop_path_uniform=args.drop_path_uniform,
            )
            student.mx_embed_layer = copy.deepcopy(mx_embed_layer)
        embed_dim = student.embed_dim
        if norm_after_he_tokenizer:
            student.he_embed_layer.norm = nn.LayerNorm(embed_dim)
            teacher.he_embed_layer.norm = nn.LayerNorm(embed_dim)
    if only_teacher:
        return teacher, teacher.embed_dim
    else:
        return student, teacher, embed_dim