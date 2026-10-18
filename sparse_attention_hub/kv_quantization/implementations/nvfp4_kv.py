"""NVFP4KV method scaffold and its configuration."""

from dataclasses import dataclass, field

from ..base import KVQuantizationConfig, PendingRecipe
from ..quantizers.implementations.nvfp4 import NVFP4QuantizerConfig
from ..registry import KVQuantizationRegistry


@dataclass(frozen=True)
class NVFP4KVConfig(KVQuantizationConfig):
    key_quantizer: NVFP4QuantizerConfig = field(default_factory=NVFP4QuantizerConfig)
    value_quantizer: NVFP4QuantizerConfig = field(default_factory=NVFP4QuantizerConfig)


@KVQuantizationRegistry.register(NVFP4KVConfig)
class NVFP4KV(PendingRecipe):
    """TODO: real/fake NVFP4 parity and hardware-specific execution."""
