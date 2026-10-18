"""ChannelBoost component and its configuration."""

from dataclasses import dataclass

from ....registry import PrecisionPolicyRegistry
from ..base import PendingPrecisionPolicy, PrecisionPolicyConfig


@dataclass(frozen=True)
class ChannelBoostConfig(PrecisionPolicyConfig):
    fraction: float = 0.125
    boosted_bits: int = 4


@PrecisionPolicyRegistry.register(ChannelBoostConfig)
class ChannelBoost(PendingPrecisionPolicy):
    """TODO: region statistics, selected channel indices and mixed-bit plans."""
