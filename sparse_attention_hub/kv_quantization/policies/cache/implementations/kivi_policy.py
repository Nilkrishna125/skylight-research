"""KIVICachePolicy buffering policy and its configuration."""

from dataclasses import dataclass

from ....registry import CachePolicyRegistry
from ..base import CachePolicyConfig, PendingCachePolicy


@dataclass(frozen=True)
class KIVICachePolicyConfig(CachePolicyConfig):
    """TODO: independent key flush/value rolling-buffer policy and phase timing."""


@CachePolicyRegistry.register(KIVICachePolicyConfig)
class KIVICachePolicy(PendingCachePolicy):
    """TODO: key group flush, independent value rolling buffer and prefill policy."""
