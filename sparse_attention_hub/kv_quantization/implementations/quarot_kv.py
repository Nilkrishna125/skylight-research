"""QuaRotKV method scaffold and its configuration."""

from dataclasses import dataclass, field

from ..base import KVQuantizationConfig, PendingRecipe
from ..quantizers.implementations.integer import IntQuantizerConfig
from ..registry import KVQuantizationRegistry
from ..transforms.implementations.random_hadamard import (
    RandomHadamardTransformConfig,
)


@dataclass(frozen=True)
class QuaRotKVConfig(KVQuantizationConfig):
    key_transform: RandomHadamardTransformConfig = field(
        default_factory=RandomHadamardTransformConfig
    )
    value_transform: RandomHadamardTransformConfig = field(
        default_factory=RandomHadamardTransformConfig
    )
    key_quantizer: IntQuantizerConfig = field(
        default_factory=lambda: IntQuantizerConfig(group_size=128)
    )
    value_quantizer: IntQuantizerConfig = field(
        default_factory=lambda: IntQuantizerConfig(group_size=128)
    )


@KVQuantizationRegistry.register(QuaRotKVConfig)
class QuaRotKV(PendingRecipe):
    """TODO: KV-only rotation experiment; not full-model QuaRot."""
