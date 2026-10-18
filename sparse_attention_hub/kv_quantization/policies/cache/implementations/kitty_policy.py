"""KittyCachePolicy buffering policy and its configuration."""

from dataclasses import dataclass

from ....registry import CachePolicyRegistry
from ..base import CachePolicyConfig, PendingCachePolicy


@dataclass(frozen=True)
class KittyCachePolicyConfig(CachePolicyConfig):
    """TODO: sink, key Q-buffer, value local/Q-buffers and page policy."""


@CachePolicyRegistry.register(KittyCachePolicyConfig)
class KittyCachePolicy(PendingCachePolicy):
    """TODO: sink/local/Q-buffer transitions and page-specific boost decisions."""
