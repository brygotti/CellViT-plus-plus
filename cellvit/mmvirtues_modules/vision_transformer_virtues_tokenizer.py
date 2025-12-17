# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from .vit_layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from einops import rearrange

logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        mx_embed_layer=PatchEmbed,
        he_embed_layer=None,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            mx_embed_layer (nn.Module): patch embedding layer for multiplex
            he_embed_layer (nn.Module): patch embedding layer for HE
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        if he_embed_layer is None:
            he_embed_layer = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            self.he_embed_layer = he_embed_layer
        
        self.he_embed_layer = he_embed_layer
        self.mx_embed_layer = mx_embed_layer
        # num_patches = self.patch_embed.num_patches
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_aligned = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_embed_unaligned = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mx_mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.he_mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed_aligned, std=0.02)
        trunc_normal_(self.pos_embed_unaligned, std=0.02)
        trunc_normal_(self.mx_mask_token, std=0.02)
        trunc_normal_(self.he_mask_token, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h, pos_embed: torch.Tensor):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return pos_embed
        pos_embed = pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)


    def prepare_tokens_with_masks(self, multiplex: list[torch.Tensor], he: list[torch.Tensor], proteins, technology=None, masks_multiplex=None, masks_he=None):
        mx_has_sample = torch.tensor([mx_sample is not None for mx_sample in multiplex])
        mx_sample_idx = torch.where(mx_has_sample)[0]
        he_has_sample = torch.tensor([he_sample is not None for he_sample in he])
        he_sample_idx = torch.where(he_has_sample)[0]
        w, h = multiplex[mx_sample_idx[0]].shape[-2:] if any(mx_has_sample) else he[he_sample_idx[0]].shape[-2:]
        dtype = multiplex[mx_sample_idx[0]].dtype if any(mx_has_sample) else he[he_sample_idx[0]].dtype

        # --- Multiplex ---
        full_mx_mask = self.mx_mask_token.to(dtype).repeat((len(multiplex), (w // self.patch_size) * (h // self.patch_size), 1)) # B HW D
        if any(mx_has_sample):
            if technology is None:
                technology = [None] * len(multiplex)
                
            multiplex = [multiplex[i] for i in mx_sample_idx]
            technology = [technology[i] for i in mx_sample_idx] if technology is not None else None
            proteins = [proteins[i] for i in mx_sample_idx]
            _, mx_ps = self.mx_embed_layer(multiplex, [None for _ in range(len(multiplex))], proteins, technology=technology) # use patch summary tokens for dino. Set HE to None
        
            mx_ps = torch.stack(mx_ps)
            mx_ps = rearrange(mx_ps, "b h w d -> b (h w) d")  # B HW D
            if masks_multiplex is not None:
                mx_ps = torch.where(masks_multiplex[mx_sample_idx].unsqueeze(-1), self.mx_mask_token.to(mx_ps.dtype).unsqueeze(0), mx_ps)
            full_mx_mask[mx_sample_idx] = mx_ps.to(dtype)
        mx_ps = full_mx_mask
        mx_ps = torch.cat((self.cls_token.expand(mx_ps.shape[0], -1, -1), mx_ps), dim=1)
        
        # --- HE ---
        full_he_mask = self.he_mask_token.to(dtype).repeat((len(he), (w // self.patch_size) * (h // self.patch_size), 1)) # B HW D
        if any(he_has_sample):
            he = torch.stack([he[i] for i in he_sample_idx])
            he = self.he_embed_layer(he)
            if masks_he is not None:
                he = torch.where(masks_he[he_sample_idx].unsqueeze(-1), self.he_mask_token.to(mx_ps.dtype).unsqueeze(0), he)

            full_he_mask[he_sample_idx] = he.to(dtype)
        he = full_he_mask


        # --- Positional embedding ---
        aligned_pos_encoding = self.interpolate_pos_encoding(mx_ps, w, h, self.pos_embed_aligned) 
        mx_ps = mx_ps + aligned_pos_encoding # Use the same pos embed for multiplex and HE
        he = he + aligned_pos_encoding[:, 1:] # mx_ps is used to assure the same interpolation (note the cls token in mx_ps vs he)

        # TODO: different positional embedding for aligned vs unaligned HE
        # if unaligned:
        #     he = he + self.pos_embed_unaligned

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    mx_ps[:, :1],   # CLS
                    self.register_tokens.expand(mx_ps.shape[0], -1, -1),
                    mx_ps[:, 1:],   # multiplex
                    he,             # HE

                ),
                dim=1,
            )
        else:
            x = torch.cat((mx_ps,he),dim=1,)

        return x
    
    def forward_features_list(self, multiplex_list, he_list, proteins_list, technology_list=None, multiplex_masks=None, he_masks=None):
        if technology_list is None:
            technology_list = [None] * len(multiplex_list)
        if multiplex_masks is None:
            multiplex_masks = [[None] * len(m) for m in multiplex_list]
        if he_masks is None:
            he_masks = [[None] * len(h) for h in he_list]
        x = [self.prepare_tokens_with_masks(multiplex, he, proteins, technology, multiplex_mask, he_mask) 
             for multiplex, he, proteins, technology, multiplex_mask, he_mask in zip(multiplex_list, he_list, proteins_list, technology_list, multiplex_masks, he_masks)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks_multiplex, masks_he in zip(all_x, multiplex_masks, he_masks):
            x_norm = self.norm(x)
            num_patches = (x.shape[1] - 1 - self.num_register_tokens) // 2
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    # "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 : self.num_register_tokens + 1 + num_patches], # multiplex
                    "x_norm_patchtokens_multiplex": x_norm[:, self.num_register_tokens + 1 : self.num_register_tokens + 1 + num_patches], # multiplex
                    "x_norm_patchtokens_he": x_norm[:, self.num_register_tokens + 1 + num_patches :], # HE
                    "x_prenorm": x,
                    "masks_multiplex": masks_multiplex,
                    "masks_he": masks_he,
                }
            )
        return output

    def forward_features(self, multiplex, he, proteins, technology=None, masks_multiplex=None, masks_he=None):
        if isinstance(masks_multiplex, list) or isinstance(masks_he, list):
            return self.forward_features_list(multiplex, he, proteins, technology, masks_multiplex, masks_he)

        x = self.prepare_tokens_with_masks(multiplex, he, proteins, technology, masks_multiplex=masks_multiplex, masks_he=masks_he)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        num_patches = (x.shape[1] - 1 - self.num_register_tokens) // 2
        return  {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens_multiplex": x_norm[:, self.num_register_tokens + 1 : self.num_register_tokens + 1 + num_patches], # multiplex
            "x_norm_patchtokens_he": x_norm[:, self.num_register_tokens + 1 + num_patches :], # HE
            "x_prenorm": x,
            "masks_multiplex": masks_multiplex,
            "masks_he": masks_he,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
