"""Plain helper functions shared by the quantizers (no shared base class)."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..policies.precision.base import PrecisionPlan
from .base import ParamDType, QuantGranularity


def check_input(
    x: torch.Tensor,
    plan: Optional[PrecisionPlan],
    *,
    allow_channel_bits: bool = False,
) -> None:
    """Reject inputs a quantizer refuses: wrong rank, non-float, NaN, unsupported plans.

    Tokens a plan keeps in full precision are handled by ``KVQuantization`` before
    quantizing, so a quantizer never receives ``keep_tokens``.
    """
    if x.ndim != 4 or not x.is_floating_point() or x.numel() == 0:
        raise ValueError(
            "Expected a nonempty floating tensor [batch, kv_heads, tokens, channels]"
        )
    if not torch.isfinite(x).all():
        raise ValueError("Cannot quantize NaN or infinity")
    if plan is not None and plan.keep_tokens is not None:
        raise ValueError("keep_tokens is applied by KVQuantization, not the quantizer")
    if plan is not None and plan.channel_bits is not None and not allow_channel_bits:
        raise NotImplementedError(
            "This quantizer does not support per-channel bits (ChannelBoost); "
            "use IntQuantizer with PER_CHANNEL granularity"
        )


@dataclass(frozen=True)
class GroupLayout:
    """How ``group_values`` rearranged a tensor, so ``ungroup_values`` can undo it."""

    granularity: QuantGranularity
    axis: int
    layout_shape: Tuple[int, ...]
    padded_length: int


def group_values(
    x: torch.Tensor, granularity: QuantGranularity, group_size: int
) -> Tuple[torch.Tensor, GroupLayout]:
    """Split ``x`` into FP32 groups of ``group_size`` values: ``[..., groups, size]``.

    PER_TOKEN groups channels, PER_CHANNEL groups tokens, and PER_BLOCK groups
    consecutive channels of one token (``group_size`` must divide head_dim). An
    incomplete last group repeats its last value, so it cannot change min/max/amax.
    """
    if granularity == QuantGranularity.PER_BLOCK and x.shape[-1] % group_size:
        raise ValueError(
            f"PER_BLOCK needs group_size ({group_size}) to divide "
            f"head_dim ({x.shape[-1]}) so that no block spans two tokens"
        )
    axis = 2 if granularity == QuantGranularity.PER_CHANNEL else 3
    if granularity == QuantGranularity.PER_BLOCK:
        data = x.detach().float().flatten(start_dim=2)
    else:
        data = x.detach().float().movedim(axis, -1)
    layout_shape = tuple(data.shape)
    padding = (-data.shape[-1]) % group_size
    if padding:
        data = torch.cat([data, data[..., -1:].expand(*data.shape[:-1], padding)], -1)
    padded_length = data.shape[-1]
    groups = data.reshape(*data.shape[:-1], padded_length // group_size, group_size)
    return groups, GroupLayout(granularity, axis, layout_shape, padded_length)


def ungroup_values(
    groups: torch.Tensor, layout: GroupLayout, shape: Tuple[int, ...]
) -> torch.Tensor:
    """Undo ``group_values``: drop padding and restore the original layout."""
    restored = groups.reshape(*layout.layout_shape[:-1], layout.padded_length)
    restored = restored[..., : layout.layout_shape[-1]]
    if layout.granularity == QuantGranularity.PER_BLOCK:
        return restored.reshape(shape)
    return restored.movedim(-1, layout.axis)


def layout_metadata(layout: GroupLayout, code_count: int) -> dict:
    """The ``QuantizedTensor.metadata`` entries that describe the grouping."""
    return {
        "layout_shape": layout.layout_shape,
        "padded_length": layout.padded_length,
        "code_count": code_count,
        "granularity": layout.granularity.value,
    }


def layout_from_metadata(metadata: dict, axis: int) -> GroupLayout:
    return GroupLayout(
        QuantGranularity(metadata["granularity"]),
        axis,
        tuple(metadata["layout_shape"]),
        metadata["padded_length"],
    )


def pack_codes(codes: torch.Tensor, bits: int) -> Tuple[torch.Tensor, int]:
    """Pack integer codes (0 .. 2**bits - 1) into uint8, lowest bits first."""
    flat = codes.to(torch.int16).flatten()
    entries_per_byte = 8 // bits
    code_count = flat.numel()
    padding = (-code_count) % entries_per_byte
    if padding:
        flat = torch.cat([flat, flat.new_zeros(padding)])
    shifts = torch.arange(entries_per_byte, device=codes.device) * bits
    packed = (flat.reshape(-1, entries_per_byte) << shifts).sum(-1).to(torch.uint8)
    return packed, code_count


def unpack_codes(packed: torch.Tensor, bits: int, code_count: int) -> torch.Tensor:
    """Undo ``pack_codes``; returns int16 codes."""
    entries_per_byte = 8 // bits
    shifts = torch.arange(entries_per_byte, device=packed.device) * bits
    flat = ((packed.to(torch.int16)[:, None] >> shifts) & (2**bits - 1)).flatten()
    return flat[:code_count]


def store_param(
    values: torch.Tensor, dtype: ParamDType, *, kind: str = "scale"
) -> torch.Tensor:
    """Round FP32 per-group parameters to their storage dtype; return them as FP32.

    Rounding is conservative so the stored range always covers the group: offsets
    (group minima) round down and scales round up, by at most one step of the
    storage dtype. Nothing is clipped; a coarser dtype only widens the range.
    A zero scale (constant group) becomes 1. A value too large for the dtype raises.
    """
    info = torch.finfo(dtype.torch_dtype)
    if values.abs().max() > info.max:
        raise ValueError(
            f"A quantization {kind} exceeds the {dtype.value} range ({info.max:g}); "
            f"use a wider {kind}_dtype"
        )
    if kind == "scale":
        values = torch.where(values == 0, torch.ones_like(values), values)
        values = values.clamp(min=info.tiny)
    stored = values.to(dtype.torch_dtype)
    if kind == "scale":
        too_far = stored.float() < values
        direction = torch.full_like(stored, float("inf"))
    else:
        too_far = stored.float() > values
        direction = torch.full_like(stored, float("-inf"))
    stored = torch.where(too_far, torch.nextafter(stored, direction), stored)
    return stored.float()


def fp_grid(exponent_bits: int, mantissa_bits: int) -> torch.Tensor:
    """Non-negative values of an OCP micro float (FP4/FP6: no inf or NaN), ascending.

    ``fp_grid(2, 1)`` is E2M1 (FP4): 0, 0.5, 1, 1.5, 2, 3, 4, 6.
    """
    bias = 2 ** (exponent_bits - 1) - 1
    values = []
    for exponent in range(2**exponent_bits):
        for mantissa in range(2**mantissa_bits):
            fraction = mantissa / 2**mantissa_bits
            if exponent == 0:  # subnormal
                values.append(fraction * 2.0 ** (1 - bias))
            else:
                values.append((1 + fraction) * 2.0 ** (exponent - bias))
    return torch.tensor(values, dtype=torch.float32)


def round_to_grid(magnitudes: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Index of the nearest grid value for non-negative inputs (clamped to the grid).

    Ties go to the even index, i.e. the value with an even mantissa: the same
    round-to-nearest-even rule hardware float conversion uses.
    """
    grid = grid.to(magnitudes.device)
    magnitudes = magnitudes.contiguous()
    midpoints = (grid[1:] + grid[:-1]) / 2
    index = torch.bucketize(magnitudes, midpoints)  # a tie returns the lower index
    at_midpoint = index < midpoints.numel()
    tie = at_midpoint & (
        magnitudes == midpoints[index.clamp(max=midpoints.numel() - 1)]
    )
    return torch.where(tie & (index % 2 == 1), index + 1, index)
