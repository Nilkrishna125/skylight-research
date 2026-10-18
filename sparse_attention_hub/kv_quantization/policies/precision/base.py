"""Choose numerical precision independently of cache buffering."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class PrecisionPlan:
    bits: int
    assignments: Optional[torch.Tensor] = None
    axis: Optional[int] = None


@dataclass(frozen=True)
class PrecisionPolicyConfig:
    pass


class PrecisionPolicy(ABC):
    def __init__(self, config: PrecisionPolicyConfig) -> None:
        self.config = config

    @abstractmethod
    def bits_for(
        self,
        layer_idx: int,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        tensor_kind: str,
        default_bits: int,
    ) -> PrecisionPlan:
        """Return a PrecisionPlan; 0 entries in future plans mean retain incoming FP dtype."""


class PendingPrecisionPolicy(PrecisionPolicy):
    def __init__(self, config: PrecisionPolicyConfig) -> None:
        super().__init__(config)
        raise NotImplementedError(
            f"{type(self).__name__} is a scaffold; mixed precision is not implemented"
        )

    def bits_for(
        self,
        layer_idx: int,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        tensor_kind: str,
        default_bits: int,
    ) -> PrecisionPlan:
        raise NotImplementedError
