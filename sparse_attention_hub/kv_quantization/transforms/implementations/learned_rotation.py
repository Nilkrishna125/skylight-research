"""LearnedRotation component and its configuration."""

from dataclasses import dataclass

from ...registry import KVTransformRegistry
from ..base import KVTransformConfig, PendingTransform


@dataclass(frozen=True)
class LearnedRotationConfig(KVTransformConfig):
    pass


@KVTransformRegistry.register(LearnedRotationConfig)
class LearnedRotation(PendingTransform):
    """TODO: calibrated (e.g. RotateKV-style), persisted and versioned rotations."""
