"""Kitty method scaffold and its configuration."""

from dataclasses import dataclass, field

from ..base import KVQuantizationConfig, PendingRecipe
from ..policies.cache.implementations.kitty_policy import KittyCachePolicyConfig
from ..policies.precision.implementations.channel_boost import ChannelBoostConfig
from ..quantizers.base import QuantGranularity
from ..quantizers.implementations.integer import IntQuantizerConfig
from ..registry import KVQuantizationRegistry


@dataclass(frozen=True)
class KittyConfig(KVQuantizationConfig):
    key_quantizer: IntQuantizerConfig = field(
        default_factory=lambda: IntQuantizerConfig(
            bits=2, granularity=QuantGranularity.PER_CHANNEL, group_size=128
        )
    )
    value_quantizer: IntQuantizerConfig = field(
        default_factory=lambda: IntQuantizerConfig(bits=2, group_size=128)
    )
    precision_policy: ChannelBoostConfig = field(default_factory=ChannelBoostConfig)
    cache_policy: KittyCachePolicyConfig = field(
        default_factory=lambda: KittyCachePolicyConfig(num_sink_tokens=32)
    )


@KVQuantizationRegistry.register(KittyConfig)
class Kitty(PendingRecipe):
    """TODO: mixed-bit pages, boost indices, sink/local/Q-buffer policies."""
