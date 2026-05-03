from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


COMPUTE_DTYPE = torch.bfloat16
EPS = 1e-12


@torch.no_grad()
def _to_fp8(x: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x.float().abs().max()
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale


def _to_col_major(x: torch.Tensor) -> torch.Tensor:
    return x.t().contiguous().t()


@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)
        return torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_inv,
            scale_b=weight_inv,
            out_dtype=input_2d.dtype,
            use_fast_accum=True,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors

        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        w_col = _to_col_major(w_fp8)
        grad_input = torch._scaled_mm(
            go_fp8,
            w_col,
            scale_a=go_inv,
            scale_b=w_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        go_t = go_fp8.t().contiguous()
        in_col = _to_col_major(in_fp8)
        grad_weight = torch._scaled_mm(
            go_t,
            in_col,
            scale_a=go_inv,
            scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )
        return grad_input, grad_weight


class Float8Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(COMPUTE_DTYPE)
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
        output = _Float8Matmul.apply(input_2d, self.weight)
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(cls, mod: nn.Linear) -> "Float8Linear":
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


class Float8LinearConfig:
    @staticmethod
    def from_recipe_name(recipe_name: str) -> "Float8LinearConfig":
        if recipe_name != "tensorwise":
            raise ValueError(f"Only 'tensorwise' recipe is supported, got {recipe_name!r}.")
        return Float8LinearConfig()


def convert_to_float8_training(
    module: nn.Module,
    *,
    config: Float8LinearConfig | None = None,
    module_filter_fn=None,
) -> nn.Module:
    del config

    def _convert(mod: nn.Module, prefix: str = "") -> None:
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float8Linear.from_float(child))

    _convert(module)
    return module


@contextmanager
def disable_fp8(module: nn.Module):
    fp8_locations: list[tuple[nn.Module, str, Float8Linear]] = []
    for name, child in module.named_modules():
        if not isinstance(child, Float8Linear):
            continue
        if "." in name:
            parent_name, attr_name = name.rsplit(".", 1)
            parent = module.get_submodule(parent_name)
        else:
            parent = module
            attr_name = name
        fp8_locations.append((parent, attr_name, child))

    if not fp8_locations:
        yield
        return

    for parent, attr_name, fp8_module in fp8_locations:
        with torch.device("meta"):
            linear = nn.Linear(
                fp8_module.in_features,
                fp8_module.out_features,
                bias=fp8_module.bias is not None,
                dtype=fp8_module.weight.dtype,
            )
        linear.weight = fp8_module.weight
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)
