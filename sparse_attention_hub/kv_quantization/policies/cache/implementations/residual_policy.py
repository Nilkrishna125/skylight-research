"""ResidualCachePolicy buffering policy and its configuration."""

from dataclasses import dataclass

from ....cache.state import TensorCacheState
from ....registry import CachePolicyRegistry
from ..base import CachePolicy, CachePolicyConfig


@dataclass(frozen=True)
class ResidualCachePolicyConfig(CachePolicyConfig):
    """Keep at least the requested FP tail; encode older complete chunks."""


@CachePolicyRegistry.register(ResidualCachePolicyConfig)
class ResidualCachePolicy(CachePolicy):
    def __init__(self, config: ResidualCachePolicyConfig) -> None:
        super().__init__(config)
        if config.num_sink_tokens < 0:
            raise ValueError("num_sink_tokens cannot be negative")
        for kind in ("key", "value"):
            if getattr(config, f"{kind}_block_tokens") < 1:
                raise ValueError("block token counts must be positive")
            if getattr(config, f"{kind}_residual_tokens") < 1:
                raise ValueError("Keep at least one recent token in FP precision")

    def next_chunk_tokens(
        self,
        tensor_kind: str,
        state: TensorCacheState,
        *,
        phase: str = "before_attention",
    ) -> int:
        if phase != "before_attention":
            return 0
        count = (
            self.config.key_block_tokens
            if tensor_kind == "key"
            else self.config.value_block_tokens
        )
        keep = (
            self.config.key_residual_tokens
            if tensor_kind == "key"
            else self.config.value_residual_tokens
        )
        return count if state.recent_tokens >= keep + count else 0
