import torch
from torch.library import get_kernel
import torch.nn.functional as F

def load_flash_attention_3():
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()

        if major != 9:
            return None
        
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kenel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    
    except Exception:
        return None

_fa3, = load_flash_attention_3()
HAS_FA3 = _fa3 is not None

_override_impl = None


