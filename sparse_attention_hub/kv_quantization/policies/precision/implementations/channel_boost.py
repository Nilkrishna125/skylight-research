"""ChannelBoost: give the most important key channels more bits (Kitty)."""

from dataclasses import dataclass
from typing import Tuple

import torch

from ....quantizers.base import KVQuantizerConfig, QuantGranularity
from ....registry import PrecisionPolicyRegistry
from ..base import PrecisionPlan, PrecisionPolicy, PrecisionPolicyConfig

KINDS = ("key", "value")


@dataclass(frozen=True)
class ChannelBoostConfig(PrecisionPolicyConfig):
    """Per chunk and per head, the ``fraction`` of channels with the largest mean
    magnitude get ``boosted_bits``; the others keep the quantizer's bits.

    Attributes:
        fraction: Share of channels boosted (Kitty: 12.5%; Kitty-Pro: 25%).
        boosted_bits: Bits of the boosted channels (2, 4 or 8).
        kinds: Which tensors are boosted; Kitty boosts keys only.
    """

    fraction: float = 0.125
    boosted_bits: int = 4
    kinds: Tuple[str, ...] = ("key",)


@PrecisionPolicyRegistry.register(ChannelBoostConfig)
class ChannelBoost(PrecisionPolicy):
    """Channels are ranked by mean |x| over the chunk's tokens, separately for each
    head, so the boost follows each chunk (Kitty's per-page boost)."""

    config: ChannelBoostConfig

    def __init__(self, config: ChannelBoostConfig) -> None:
        super().__init__(config)
        if not 0 < config.fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        if config.boosted_bits not in (2, 4, 8):
            raise ValueError("boosted_bits must be 2, 4 or 8")
        if any(kind not in KINDS for kind in config.kinds):
            raise ValueError(f"kinds must be drawn from {KINDS}")

    def bits_for(
        self,
        layer_idx: int,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        tensor_kind: str,
        default_bits: int,
    ) -> PrecisionPlan:
        if tensor_kind not in self.config.kinds:
            return PrecisionPlan(bits=default_bits)
        channels = x.shape[-1]
        boosted = max(1, round(self.config.fraction * channels))
        importance = x.abs().float().mean(dim=-2)  # [batch, heads, channels]
        top = importance.topk(boosted, dim=-1).indices
        channel_bits = torch.full(
            importance.shape, default_bits, dtype=torch.int64, device=x.device
        )
        channel_bits.scatter_(-1, top, self.config.boosted_bits)
        return PrecisionPlan(bits=default_bits, channel_bits=channel_bits)

    def validate_quantizers(
        self, key_quantizer: KVQuantizerConfig, value_quantizer: KVQuantizerConfig
    ) -> None:
        from ....quantizers.implementations.integer import IntQuantizerConfig

        for kind, quantizer in zip(KINDS, (key_quantizer, value_quantizer)):
            if kind not in self.config.kinds:
                continue
            if not isinstance(quantizer, IntQuantizerConfig) or (
                QuantGranularity(quantizer.granularity) != QuantGranularity.PER_CHANNEL
            ):
                raise ValueError(
                    f"ChannelBoost on {kind}s needs an IntQuantizer with PER_CHANNEL "
                    "granularity, so each group has one bit width"
                )
