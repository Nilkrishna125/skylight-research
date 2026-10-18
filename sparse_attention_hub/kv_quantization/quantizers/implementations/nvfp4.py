"""NVFP4Quantizer component and its configuration."""

from dataclasses import dataclass

from ...registry import KVQuantizerRegistry
from ..base import KVQuantizerConfig, PendingQuantizer, ScalarQuantizer


@dataclass(frozen=True)
class NVFP4QuantizerConfig(KVQuantizerConfig):
    bits: int = 4
    group_size: int = 16


@KVQuantizerRegistry.register(NVFP4QuantizerConfig)
class NVFP4Quantizer(PendingQuantizer, ScalarQuantizer):
    """TODO: E2M1 payload, E4M3 block scales, FP32 global scale and packing."""
