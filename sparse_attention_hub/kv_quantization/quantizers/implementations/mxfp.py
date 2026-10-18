"""MXFPQuantizer component and its configuration."""

from dataclasses import dataclass

from ...registry import KVQuantizerRegistry
from ..base import KVQuantizerConfig, PendingQuantizer, ScalarQuantizer


@dataclass(frozen=True)
class MXFPQuantizerConfig(KVQuantizerConfig):
    format: str = "E2M1"


@KVQuantizerRegistry.register(MXFPQuantizerConfig)
class MXFPQuantizer(PendingQuantizer, ScalarQuantizer):
    """TODO: format-dependent codes, E8M0 scales and block layout."""
