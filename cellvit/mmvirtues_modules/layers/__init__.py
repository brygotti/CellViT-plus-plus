from mmvirtues_dataset_loading.utils.utils import is_rank0
from loguru import logger

try:
    import xformers
    from .transformer_xformers import MarkerAttentionEncoderBlock, ChannelAttentionEncoderBlock, FullAttentionEncoderBlock, PatchAttentionBlock
    from .mask_utils_xformers import SA_BIAS_CACHE
    USE_XFORMERS = True
    USE_FLASHATTENTION = False
    if is_rank0():
        logger.info(f'Using xformers for FlexDualVirTues')
    
except ImportError:
    from .transformers_flashattention import MarkerAttentionEncoderBlock, ChannelAttentionEncoderBlock, FullAttentionEncoderBlock, PatchAttentionBlock
    from .mask_utils_flashattention import SA_BIAS_CACHE
    USE_XFORMERS = False
    USE_FLASHATTENTION = True
    if is_rank0():
        logger.info(f'Using flash attention for FlexDualVirTues')