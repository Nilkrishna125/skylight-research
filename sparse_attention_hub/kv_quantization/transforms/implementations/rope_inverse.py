"""RoPEInverse component and its configuration."""

from dataclasses import dataclass

from ...registry import KVTransformRegistry
from ..base import KVTransformConfig, PendingTransform


@dataclass(frozen=True)
class RoPEInverseConfig(KVTransformConfig):
    pass


@KVTransformRegistry.register(RoPEInverseConfig)
class RoPEInverse(PendingTransform):
    """TODO: model-specific RoPE context and absolute positions; reapply on read."""
