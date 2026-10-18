"""TurboQuantKV method scaffold and its configuration."""

from dataclasses import dataclass, field

from ..base import KVQuantizationConfig, PendingRecipe
from ..quantizers.implementations.non_uniform import NonUniformQuantizerConfig
from ..registry import KVQuantizationRegistry
from ..transforms.implementations.random_rotation import RandomRotationConfig


@dataclass(frozen=True)
class TurboQuantKVConfig(KVQuantizationConfig):
    """MSE variant only; a QJL inner-product estimator needs another contract."""

    key_transform: RandomRotationConfig = field(default_factory=RandomRotationConfig)
    value_transform: RandomRotationConfig = field(default_factory=RandomRotationConfig)
    key_quantizer: NonUniformQuantizerConfig = field(
        default_factory=NonUniformQuantizerConfig
    )
    value_quantizer: NonUniformQuantizerConfig = field(
        default_factory=NonUniformQuantizerConfig
    )


@KVQuantizationRegistry.register(TurboQuantKVConfig)
class TurboQuantKV(PendingRecipe):
    """TODO: normalized rotation + MSE codebooks + norm restoration."""
