"""Transform contract, base configuration and pending implementation helper."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple, cast

import torch

from ..context import AttentionContext
from ..registry import KVTransformRegistry


@dataclass(frozen=True)
class KVTransformConfig:
    pass


class KVTransform(ABC):
    def __init__(self, config: KVTransformConfig) -> None:
        self.config = config

    @abstractmethod
    def forward(
        self, x: torch.Tensor, context: AttentionContext
    ) -> Tuple[torch.Tensor, Any]:
        """Return transformed data and all chunk-specific state needed by inverse."""

    @abstractmethod
    def inverse(
        self, x: torch.Tensor, context: AttentionContext, state: Any
    ) -> torch.Tensor:
        """Restore the original coordinate system, including RoPE when applicable."""

    def calibrate(self, samples: Any) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} has no calibration implementation"
        )

    @classmethod
    def create_from_config(cls, config: KVTransformConfig) -> "KVTransform":
        return cast(KVTransform, KVTransformRegistry.create(config))


class PendingTransform(KVTransform):
    def __init__(self, config: KVTransformConfig) -> None:
        super().__init__(config)
        raise NotImplementedError(
            f"{type(self).__name__} is a scaffold; implement before selecting it"
        )

    def forward(
        self, x: torch.Tensor, context: AttentionContext
    ) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError

    def inverse(
        self, x: torch.Tensor, context: AttentionContext, state: Any
    ) -> torch.Tensor:
        raise NotImplementedError
