"""FP8Quantizer component and its configuration."""

from dataclasses import dataclass

from ...registry import KVQuantizerRegistry
from ..base import KVQuantizerConfig, PendingQuantizer, ScalarQuantizer


@dataclass(frozen=True)
class FP8QuantizerConfig(KVQuantizerConfig):
    bits: int = 8
    format: str = "E4M3"


@KVQuantizerRegistry.register(FP8QuantizerConfig)
class FP8Quantizer(PendingQuantizer, ScalarQuantizer):
    """TODO: format-specific rounding, clipping, scale and storage."""
