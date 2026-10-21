"""IntQuantizer component and its configuration."""

from dataclasses import dataclass
from typing import Optional

import torch

from ...cache.block import QuantizedTensor
from ...policies.precision.base import PrecisionPlan
from ...registry import KVQuantizerRegistry
from ..base import KVQuantizer, KVQuantizerConfig, ParamDType, QuantGranularity


@dataclass(frozen=True)
class IntQuantizerConfig(KVQuantizerConfig):
    pass


@KVQuantizerRegistry.register(IntQuantizerConfig)
class IntQuantizer(KVQuantizer):
    """FP32 group statistics and uint8-packed codes, on CPU or CUDA.

    PER_TOKEN groups channels, PER_CHANNEL groups tokens, and PER_BLOCK groups
    consecutive channels of one token (``group_size`` must divide head_dim, so a
    block never spans two tokens). Padding repeats the last entry so it cannot
    change a group's min/max.
    """

    def __init__(self, config: IntQuantizerConfig) -> None:
        super().__init__(config)
        if config.bits not in (2, 4, 8):
            raise ValueError("IntQuantizer supports 2, 4 or 8 bits")
        if config.group_size < 1:
            raise ValueError("group_size must be positive")
        self.granularity = QuantGranularity(config.granularity)

    def quantize(
        self, x: torch.Tensor, plan: Optional[PrecisionPlan] = None
    ) -> QuantizedTensor:
        if x.ndim != 4 or not x.is_floating_point() or x.numel() == 0:
            raise ValueError(
                "Expected a nonempty floating tensor [batch, kv_heads, tokens, channels]"
            )
        if not torch.isfinite(x).all():
            raise ValueError("Cannot quantize NaN or infinity")
        if plan is not None and plan.assignments is not None:
            raise NotImplementedError(
                "Mixed-bit packing is not implemented; use FixedPrecision"
            )
        bits = self.config.bits if plan is None else plan.bits
        if bits not in (2, 4, 8):
            raise ValueError("INT precision plan must use 2, 4 or 8 bits")

        if (
            self.granularity == QuantGranularity.PER_BLOCK
            and x.shape[-1] % self.config.group_size
        ):
            raise ValueError(
                f"PER_BLOCK needs group_size ({self.config.group_size}) to divide "
                f"head_dim ({x.shape[-1]}) so that no block spans two tokens"
            )
        axis = 2 if self.granularity == QuantGranularity.PER_CHANNEL else 3
        if self.granularity == QuantGranularity.PER_BLOCK:
            data = x.detach().float().flatten(start_dim=2)
        else:
            data = x.detach().float().movedim(axis, -1)
        layout_shape = tuple(data.shape)
        size = self.config.group_size
        padding = (-data.shape[-1]) % size
        if padding:
            data = torch.cat(
                [data, data[..., -1:].expand(*data.shape[:-1], padding)], -1
            )
        padded_length = data.shape[-1]
        groups = data.reshape(*data.shape[:-1], padded_length // size, size)
        lo = groups.amin(dim=-1, keepdim=True)
        hi = groups.amax(dim=-1, keepdim=True)

        # Statistics are FP32. Offsets and scales are rounded to their storage
        # dtypes *before* computing codes, so decoding uses exactly what is stored.
        offset: Optional[torch.Tensor]
        if self.config.symmetric:
            # Signed symmetric [-qmax, qmax], stored as unsigned codes; no offset.
            qmax = 2 ** (bits - 1) - 1
            scale = _store(
                torch.maximum(lo.abs(), hi.abs()) / qmax, self.config.scale_dtype
            )
            offset = None
            codes = (groups / scale).round().clamp(-qmax, qmax) + qmax
        else:
            offset = _store(lo, self.config.offset_dtype, kind="offset")
            scale = _store((hi - offset) / (2**bits - 1), self.config.scale_dtype)
            # A constant group decodes to its offset (exact when the offset dtype
            # represents the value).
            codes = ((groups - offset) / scale).round().clamp(0, 2**bits - 1)

        flat = codes.to(torch.int16).flatten()
        entries_per_byte = 8 // bits
        code_count = flat.numel()
        pack_padding = (-code_count) % entries_per_byte
        if pack_padding:
            flat = torch.cat([flat, flat.new_zeros(pack_padding)])
        shifts = torch.arange(entries_per_byte, device=x.device) * bits
        packed = (flat.reshape(-1, entries_per_byte) << shifts).sum(-1).to(torch.uint8)
        return QuantizedTensor(
            codes=packed,
            scales=scale.to(self.config.scale_dtype.torch_dtype).contiguous(),
            # Floating offset (the group minimum), not an integer zero code.
            zero_points=(
                None
                if offset is None
                else offset.to(self.config.offset_dtype.torch_dtype).contiguous()
            ),
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=axis,
            bits=bits,
            format_id="uniform_int_v1",
            group_size=size,
            metadata={
                "layout_shape": layout_shape,
                "padded_length": padded_length,
                "code_count": code_count,
                "granularity": self.granularity.value,
                "symmetric": self.config.symmetric,
            },
        )

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        if q.format_id != "uniform_int_v1":
            raise ValueError(f"Unsupported integer encoding: {q.format_id}")
        entries_per_byte = 8 // q.bits
        shifts = torch.arange(entries_per_byte, device=q.codes.device) * q.bits
        flat = (
            (q.codes.to(torch.int16)[:, None] >> shifts) & (2**q.bits - 1)
        ).flatten()
        flat = flat[: q.metadata["code_count"]].float()
        layout_shape = q.metadata["layout_shape"]
        padded = q.metadata["padded_length"]
        groups = flat.reshape(*layout_shape[:-1], padded // q.group_size, q.group_size)
        scales = q.scales.float()
        if q.metadata["symmetric"]:
            restored = (groups - (2 ** (q.bits - 1) - 1)) * scales
        else:
            assert q.zero_points is not None
            restored = groups * scales + q.zero_points.float()
        restored = restored.reshape(*layout_shape[:-1], padded)[..., : layout_shape[-1]]
        if q.metadata["granularity"] == QuantGranularity.PER_BLOCK.value:
            restored = restored.reshape(q.shape)
        else:
            restored = restored.movedim(-1, q.axis)
        return restored.to(device=q.device, dtype=q.dtype).contiguous()


def _store(
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
