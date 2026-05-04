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


def _resolve_use_fa3():
    if _override_impl == "fa3":
        assert HAS_FA3, "Cannot override to FA3: not avialble on this hardware"
        return True
    if _override_impl == "sdpa":
        return False
    
    if HAS_FA3:
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False
        
    return False


USE_FA3 = _resolve_use_fa3()
