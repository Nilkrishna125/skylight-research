"""OutlierTokenPolicy component and its configuration."""

from dataclasses import dataclass

from ....registry import PrecisionPolicyRegistry
from ..base import PendingPrecisionPolicy, PrecisionPolicyConfig


@dataclass(frozen=True)
class OutlierTokenPolicyConfig(PrecisionPolicyConfig):
    threshold: float = 6.0


@PrecisionPolicyRegistry.register(OutlierTokenPolicyConfig)
class OutlierTokenPolicy(PendingPrecisionPolicy):
    """TODO: token selection; per-element outliers require a separate extension."""
