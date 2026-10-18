"""IntQuantizer component and its configuration."""

from dataclasses import dataclass
from typing import Optional

import torch

from ...cache.block import QuantizedTensor
from ...policies.precision.base import PrecisionPlan
from ...registry import KVQuantizerRegistry
from ..base import KVQuantizerConfig, QuantGranularity, ScalarQuantizer


@dataclass(frozen=True)
class IntQuantizerConfig(KVQuantizerConfig):
    pass


@KVQuantizerRegistry.register(IntQuantizerConfig)
class IntQuantizer(ScalarQuantizer):
    """FP32 group statistics and uint8-packed codes, on CPU or CUDA.

    PER_TOKEN groups channels, PER_CHANNEL groups tokens, and PER_BLOCK groups
    flattened token/channel entries independently within each batch/KV head.
    Padding repeats the last entry so it cannot change a group's min/max.
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

        if self.config.symmetric:
            # Signed symmetric [-qmax, qmax], stored as unsigned codes.
            qmax = 2 ** (bits - 1) - 1
            scale = torch.maximum(lo.abs(), hi.abs()) / qmax
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            offset = -qmax * scale
            codes = (groups / scale).round().clamp(-qmax, qmax) + qmax
        else:
            scale = (hi - lo) / (2**bits - 1)
            # Constant groups reconstruct exactly using code=0, offset=lo.
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            offset = lo
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
            scales=scale.contiguous(),
            zero_points=offset.contiguous(),  # Floating offset, not integer zero code.
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
        restored = groups * q.scales + q.zero_points
        restored = restored.reshape(*layout_shape[:-1], padded)[..., : layout_shape[-1]]
        if q.metadata["granularity"] == QuantGranularity.PER_BLOCK.value:
            restored = restored.reshape(q.shape)
        else:
            restored = restored.movedim(-1, q.axis)
        return restored.to(device=q.device, dtype=q.dtype).contiguous()
