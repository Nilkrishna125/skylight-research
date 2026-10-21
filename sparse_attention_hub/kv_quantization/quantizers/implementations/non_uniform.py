"""NonUniformQuantizer component and its configuration."""

from dataclasses import dataclass
from typing import Optional

from ...registry import KVQuantizerRegistry
from ..base import KVQuantizerConfig, PendingQuantizer


@dataclass(frozen=True)
class NonUniformQuantizerConfig(KVQuantizerConfig):
    codebook_path: Optional[str] = None


@KVQuantizerRegistry.register(NonUniformQuantizerConfig)
class NonUniformQuantizer(PendingQuantizer):
    """TODO: codebook construction/loading, indexing, outliers and norms."""
