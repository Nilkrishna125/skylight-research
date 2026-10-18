"""KIVI method scaffold and its configuration."""

from dataclasses import dataclass, field

from ..base import KVQuantizationConfig, PendingRecipe
from ..policies.cache.implementations.kivi_policy import KIVICachePolicyConfig
from ..quantizers.base import QuantGranularity
from ..quantizers.implementations.integer import IntQuantizerConfig
from ..registry import KVQuantizationRegistry


@dataclass(frozen=True)
class KIVIConfig(KVQuantizationConfig):
    key_quantizer: IntQuantizerConfig = field(
        default_factory=lambda: IntQuantizerConfig(
            bits=2, granularity=QuantGranularity.PER_CHANNEL
        )
    )
    value_quantizer: IntQuantizerConfig = field(
        default_factory=lambda: IntQuantizerConfig(bits=2)
    )
    cache_policy: KIVICachePolicyConfig = field(default_factory=KIVICachePolicyConfig)


@KVQuantizationRegistry.register(KIVIConfig)
class KIVI(PendingRecipe):
    """TODO: faithful KIVI grouping, K/V buffers and phase timing."""
