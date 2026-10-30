"""IntQuantizer component and its configuration."""

from dataclasses import dataclass
from typing import Optional

import torch

from ...cache.block import QuantizedTensor
from ...context import AttentionContext
from ...policies.precision.base import PrecisionPlan
from ...registry import KVQuantizerRegistry
from ..base import KVQuantizer, KVQuantizerConfig, QuantGranularity
from ..common import (
    check_input,
    group_values,
    layout_from_metadata,
    layout_metadata,
    pack_codes,
    store_param,
    ungroup_values,
    unpack_codes,
)


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

    A plan's ``channel_bits`` (mixed precision, e.g. ChannelBoost) needs PER_CHANNEL
    groups, so every group has one bit width. Codes of each width are packed
    separately; one byte per group records its width.
    """

    def __init__(self, config: IntQuantizerConfig) -> None:
        super().__init__(config)
        if config.bits not in (2, 4, 8):
            raise ValueError("IntQuantizer supports 2, 4 or 8 bits")
        if config.group_size < 1:
            raise ValueError("group_size must be positive")
        self.granularity = QuantGranularity(config.granularity)

    def quantize(
        self,
        x: torch.Tensor,
        plan: Optional[PrecisionPlan] = None,
        *,
        context: Optional[AttentionContext] = None,
    ) -> QuantizedTensor:
        check_input(x, plan, allow_channel_bits=True)
        bits = self.config.bits if plan is None else plan.bits
        if bits not in (2, 4, 8):
            raise ValueError("INT precision plan must use 2, 4 or 8 bits")
        if plan is not None and plan.channel_bits is not None:
            return self._quantize_mixed(x, plan.channel_bits, bits)
        groups, layout = group_values(x, self.granularity, self.config.group_size)
        lo = groups.amin(dim=-1, keepdim=True)
        hi = groups.amax(dim=-1, keepdim=True)

        # Statistics are FP32. Offsets and scales are rounded to their storage
        # dtypes *before* computing codes, so decoding uses exactly what is stored.
        offset: Optional[torch.Tensor]
        if self.config.symmetric:
            # Signed symmetric [-qmax, qmax], stored as unsigned codes; no offset.
            qmax = 2 ** (bits - 1) - 1
            scale = store_param(
                torch.maximum(lo.abs(), hi.abs()) / qmax, self.config.scale_dtype
            )
            offset = None
            codes = (groups / scale).round().clamp(-qmax, qmax) + qmax
        else:
            offset = store_param(lo, self.config.offset_dtype, kind="offset")
            scale = store_param((hi - offset) / (2**bits - 1), self.config.scale_dtype)
            # A constant group decodes to its offset (exact when the offset dtype
            # represents the value).
            codes = ((groups - offset) / scale).round().clamp(0, 2**bits - 1)

        packed, code_count = pack_codes(codes, bits)
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
            axis=layout.axis,
            bits=bits,
            format_id="uniform_int_v1",
            group_size=self.config.group_size,
            metadata={
                **layout_metadata(layout, code_count),
                "symmetric": self.config.symmetric,
            },
        )

    def _quantize_mixed(
        self, x: torch.Tensor, channel_bits: torch.Tensor, default_bits: int
    ) -> QuantizedTensor:
        """Per-channel bit widths: every PER_CHANNEL group has its channel's width."""
        if self.granularity != QuantGranularity.PER_CHANNEL:
            raise ValueError(
                "Per-channel bits need PER_CHANNEL granularity, so each group has "
                "one bit width"
            )
        if tuple(channel_bits.shape) != (x.shape[0], x.shape[1], x.shape[3]):
            raise ValueError("channel_bits must have shape [batch, heads, head_dim]")
        widths = sorted(int(w) for w in channel_bits.unique())
        if any(w not in (2, 4, 8) for w in widths):
            raise ValueError("Per-channel bits must be 2, 4 or 8")
        groups, layout = group_values(x, self.granularity, self.config.group_size)
        # groups: [batch, heads, channels, n_groups, size]; one width per channel.
        group_bits = channel_bits.to(groups.device)[..., None].expand(groups.shape[:-1])
        bits = group_bits[..., None].float()
        lo = groups.amin(dim=-1, keepdim=True)
        hi = groups.amax(dim=-1, keepdim=True)
        offset: Optional[torch.Tensor]
        if self.config.symmetric:
            qmax = 2 ** (bits - 1) - 1
            scale = store_param(
                torch.maximum(lo.abs(), hi.abs()) / qmax, self.config.scale_dtype
            )
            offset = None
            codes = torch.minimum(torch.maximum((groups / scale).round(), -qmax), qmax)
            codes = codes + qmax
        else:
            levels = 2**bits - 1
            offset = store_param(lo, self.config.offset_dtype, kind="offset")
            scale = store_param((hi - offset) / levels, self.config.scale_dtype)
            codes = torch.minimum(((groups - offset) / scale).round().clamp(min=0), levels)

        flat_bits = group_bits.flatten()
        flat_codes = codes.reshape(-1, groups.shape[-1])
        parts, counts, byte_counts = [], [], []
        for width in widths:
            packed, count = pack_codes(flat_codes[flat_bits == width], width)
            parts.append(packed)
            counts.append(count)
            byte_counts.append(packed.numel())
        return QuantizedTensor(
            codes=torch.cat(parts),
            scales=scale.to(self.config.scale_dtype.torch_dtype).contiguous(),
            zero_points=(
                None
                if offset is None
                else offset.to(self.config.offset_dtype.torch_dtype).contiguous()
            ),
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=layout.axis,
            bits=default_bits,
            format_id="uniform_int_v1",
            group_size=self.config.group_size,
            metadata={
                **layout_metadata(layout, flat_codes.numel()),
                "symmetric": self.config.symmetric,
                "widths": widths,
                "width_code_counts": counts,
                "width_byte_counts": byte_counts,
            },
            group_bits=flat_bits.to(torch.uint8).contiguous(),
        )

    def _unpack_mixed(self, q: QuantizedTensor) -> torch.Tensor:
        assert q.group_bits is not None
        flat_bits = q.group_bits.long()
        codes = torch.zeros(
            flat_bits.numel(), q.group_size, dtype=torch.float32, device=q.codes.device
        )
        start = 0
        for width, count, nbytes in zip(
            q.metadata["widths"],
            q.metadata["width_code_counts"],
            q.metadata["width_byte_counts"],
        ):
            part = unpack_codes(q.codes[start : start + nbytes], width, count)
            codes[flat_bits == width] = part.float().reshape(-1, q.group_size)
            start += nbytes
        return codes

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        if q.format_id != "uniform_int_v1":
            raise ValueError(f"Unsupported integer encoding: {q.format_id}")
        layout = layout_from_metadata(q.metadata, q.axis)
        if q.group_bits is not None:
            flat = self._unpack_mixed(q)
            group_bits = q.group_bits.float().reshape(q.scales.shape)
        else:
            flat = unpack_codes(q.codes, q.bits, q.metadata["code_count"]).float()
            group_bits = None
        groups = flat.reshape(
            *layout.layout_shape[:-1], layout.padded_length // q.group_size, q.group_size
        )
        scales = q.scales.float()
        if q.metadata["symmetric"]:
            bits = q.bits if group_bits is None else group_bits
            restored = (groups - (2 ** (bits - 1) - 1)) * scales
        else:
            assert q.zero_points is not None
            restored = groups * scales + q.zero_points.float()
        restored = ungroup_values(restored, layout, q.shape)
        return restored.to(device=q.device, dtype=q.dtype).contiguous()
