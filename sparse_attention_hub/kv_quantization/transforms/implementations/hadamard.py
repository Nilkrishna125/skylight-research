"""HadamardTransform component and its configuration."""

from dataclasses import dataclass

from ...registry import KVTransformRegistry
from ..base import KVTransformConfig, PendingTransform


@dataclass(frozen=True)
class HadamardTransformConfig(KVTransformConfig):
    pass


@KVTransformRegistry.register(HadamardTransformConfig)
class HadamardTransform(PendingTransform):
    """TODO: normalized Hadamard transform, dimension validation and inverse."""
