"""Per-layer state with independent key and value histories."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .block import StoredChunk


@dataclass
class TensorCacheState:
    sink: Optional[torch.Tensor] = None
    recent: Optional[torch.Tensor] = None
    chunks: List[StoredChunk] = field(default_factory=list)

    @property
    def compressed_tokens(self) -> int:
        return sum(chunk.token_count for chunk in self.chunks)

    @property
    def recent_tokens(self) -> int:
        return 0 if self.recent is None else self.recent.shape[-2]

    @property
    def sink_tokens(self) -> int:
        return 0 if self.sink is None else self.sink.shape[-2]


@dataclass
class LayerKVState:
    seen_tokens: int = 0
    key: TensorCacheState = field(default_factory=TensorCacheState)
    value: TensorCacheState = field(default_factory=TensorCacheState)
    key_schema: Optional[Tuple[Any, ...]] = None
    value_schema: Optional[Tuple[Any, ...]] = None
    method_state: Dict[str, Any] = field(default_factory=dict)
