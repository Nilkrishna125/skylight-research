"""Choose when key and value regions are compressed."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...cache.state import TensorCacheState

if TYPE_CHECKING:
    from ...cache.store import QuantizedKVStore


@dataclass(frozen=True)
class CachePolicyConfig:
    """Temporal policy, separate from choosing numerical precision."""

    num_sink_tokens: int = 0
    key_block_tokens: int = 64
    value_block_tokens: int = 64
    key_residual_tokens: int = 128
    value_residual_tokens: int = 128


class CachePolicy(ABC):
    def __init__(self, config: CachePolicyConfig) -> None:
        self.config = config

    @abstractmethod
    def next_chunk_tokens(
        self,
        tensor_kind: str,
        state: TensorCacheState,
        *,
        phase: str = "before_attention",
    ) -> int:
        """Return how much of the recent prefix can be encoded now."""

    def after_attention(self, store: "QuantizedKVStore", layer_idx: int) -> None:
        """Extension point for methods that commit compression after attention."""


class PendingCachePolicy(CachePolicy):
    def __init__(self, config: CachePolicyConfig) -> None:
        super().__init__(config)
        raise NotImplementedError(
            f"{type(self).__name__} is a scaffold; its buffer/phase policy is not implemented"
        )

    def next_chunk_tokens(
        self,
        tensor_kind: str,
        state: TensorCacheState,
        *,
        phase: str = "before_attention",
    ) -> int:
        raise NotImplementedError
