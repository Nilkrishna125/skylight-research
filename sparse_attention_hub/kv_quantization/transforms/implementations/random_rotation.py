"""RandomRotation component and its configuration."""

from dataclasses import dataclass

from ...registry import KVTransformRegistry
from ..base import KVTransformConfig, PendingTransform


@dataclass(frozen=True)
class RandomRotationConfig(KVTransformConfig):
    seed: int = 0


@KVTransformRegistry.register(RandomRotationConfig)
class RandomRotation(PendingTransform):
    """TODO: stable per-layer/head orthogonal matrices and norm conventions."""
