"""FakeNVFP4Quantizer component and its configuration."""

from dataclasses import dataclass

from ...registry import KVQuantizerRegistry
from ..base import PendingQuantizer
from .nvfp4 import NVFP4QuantizerConfig


@dataclass(frozen=True)
class FakeNVFP4QuantizerConfig(NVFP4QuantizerConfig):
    pass


@KVQuantizerRegistry.register(FakeNVFP4QuantizerConfig)
class FakeNVFP4Quantizer(PendingQuantizer):
    """TODO: match the NVFP4 codec's numbers without native FP4 hardware."""
