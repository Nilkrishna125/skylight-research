"""KVQuantNUQ method scaffold and its configuration."""

from dataclasses import dataclass, field

from ..base import KVQuantizationConfig, PendingRecipe
from ..quantizers.base import QuantGranularity
from ..quantizers.implementations.non_uniform import NonUniformQuantizerConfig
from ..registry import KVQuantizationRegistry
from ..transforms.implementations.rope_inverse import RoPEInverseConfig


@dataclass(frozen=True)
class KVQuantNUQConfig(KVQuantizationConfig):
    key_transform: RoPEInverseConfig = field(default_factory=RoPEInverseConfig)
    key_quantizer: NonUniformQuantizerConfig = field(
        default_factory=lambda: NonUniformQuantizerConfig(
            granularity=QuantGranularity.PER_CHANNEL
        )
    )
    value_quantizer: NonUniformQuantizerConfig = field(
        default_factory=NonUniformQuantizerConfig
    )


@KVQuantizationRegistry.register(KVQuantNUQConfig)
class KVQuantNUQ(PendingRecipe):
    """TODO: calibrated codebooks, pre-RoPE keys and per-element outliers."""
