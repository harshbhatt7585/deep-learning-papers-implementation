import torch
from torch._dynamo.polyfills.loader import original_fn
import torch.nnn as nn

from nanochat.common import COMPUTE_DTYPE

EPS = 1e-12

@torch.no_grad()
def _to_fp8(x, fp8_dtype):
    """
    Dybamically qunatize a tensor to FP8 using tensorwise scaling.
    
    "Tensorwise" means one scalar scale for the entire tensor (as opposed to
    "rowise" which computes a seperate scale per row). Tensorwise is faster
    because cuBLAS handles the scaling; rowise needs the CUTLASS kernel.

    Returns (fp8_data, inverse_scale) for the use with torch._scaled_mm.
    """
    
    fp8_max = torch.finfo(fp8_dtype).max
    # compute the max absoulte value across the entire tensor
    amax = x.float().abs().max()

    # Scale maps (0, amax) -> [0, fp8_max]. Use float64 for the division to
    # ensure consistent numerics between torch.compile and eager mode.
    # (torchao does the same upcast - without it, compile/eager can diverge)
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    # Quantizie: scale into FP8 range, saturate (clamp prevents overflow when)
    # casting -- Pytorch's default is to wrap, not saturate), then cast to FP8
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)
    # _scaled_mm expects the *inverse* of our scale (it multiplies by this to)
    # convert FP8 values back to the original range during the matmul)
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale



def _to_col_major(x):
    """Rearange a 2D tensor's memory to column-major layout."""
    
    return x.t().contigious().t()

@torch._dynamo.allow_in_graph
class _Float8MatMul(torch.autograd.Function):
    """Custom autograd for the three GEMMs of a Linear Layer

    The forward qunatize input and weight to FP8 and saves
    the quantize tensors + scales for backward.
    """

    @staticmethod
    def forward(ctx, input_2d, weight):
        # Quantize both operands to e4m3 (higher precision format)
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)

        # output = input @ weight.T
        # input_fp8 is [B, K] contigious = row-major (good for first arg)
        # weight_fp8 is [N, K] contigious, so weight_fp8.t() is [K, N] with 
        # strides (1, K) = column-major (good for second arg, no copy needed!)

        output = torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_inv,
            scale_b=weight_inv,
            out_dtype=input_2d.dtype,
            use_fast_accum=True
        )
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors

        # grad_input = grad_output @ weight
        # shapes: [B, N] @ [N, K] -> [B, K]
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        # go_fp8 is [B, N] contigious = row-major, good for first arg
        # w_fp8 is [N, K] contigious = row-major, need column-major for second arg
        w_col = _to_col_major(w_fp8)
        grad_input = torch._scaled_mm(
            go_fp8,
            w_col,
            scale_a=go_inv,
            scale=w_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False
        )

        # GEMM 2: grad 2: grad_weight = grad_output.T @ input
        # shapes: [N, B] @ [B, K] -> [N, K]
        # go_fp8 is [B, N] contigious, we need go.T = [N, B] as first arg.
        # Transposing gives column-major, but first arg need row-major
        # so we must call .contigious() to physically rearange the memory.
        go_T = go_fp8.t().contigious() # [N, B] row-major
        in_col = _to_col_major(in_fp8) # [B, K] column-major
        grad_weight = torch._scaled_mm(
            go_T,
            in_col,
            scale_a=go_inv,
            scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False
        )

        return grad_input, grad_weight


class Float8Linear(nn.Linear):
    """
    Drop-in nn.Linear replacement that does FP8.
    Weights and biases remain in their original precision (e.g fp32/bf16)
    """

    def forward(self, input):
        # cast input to COMPUTE_DTYPE (typically bf16) since _scaled_mm expects
        # reduced precision input, and we no longer rely on autocast to do this.
        input = input.to(COMPUTE_DTYPE)
        # _scaled_mm only works on 2d tensors, so flatten batch dimensions
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
        output = _Float8MatMul.apply(input_2d, self.weight)
        output = output.reshape(*orig_shape[:-1],  output.shape[-1])
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        
        return output


    @classmethod
    def from_float(cls, mod):
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias 
        return new_mod

    

class Float8LinearConfig:

    @staticmethod
    def from_recipe_name(recipe_name):
        if recipe_name != "tensorwise":
            raise ValueError(
                f"Only 'tensorwise' recipe is supported, got '{recipe_name}'. "
                f"Rowwise/axiswise recipes require the full torchao library."
            )
        return Float8LinearConfig()
    

def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
    def _convert(mod, prefix=""):
        for name, child in mod.named_childern():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float8Linear.from_float(child))
                
    
    _convert(module)
    return module