"""Quantizer contracts, grouping and base configuration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, cast

import torch

from ..cache.block import QuantizedTensor
from ..policies.precision.base import PrecisionPlan
from ..registry import KVQuantizerRegistry


class QuantGranularity(str, Enum):
    PER_TOKEN = "per_token"
    PER_CHANNEL = "per_channel"
    PER_BLOCK = "per_block"


@dataclass(frozen=True)
class KVQuantizerConfig:
    bits: int = 4
    granularity: QuantGranularity = QuantGranularity.PER_TOKEN
    group_size: int = 32
    symmetric: bool = False


class KVQuantizer(ABC):
    def __init__(self, config: KVQuantizerConfig) -> None:
        self.config = config

    @abstractmethod
    def quantize(
        self, x: torch.Tensor, plan: Optional[PrecisionPlan] = None
    ) -> QuantizedTensor:
        """Encode with the supplied precision plan, including layout metadata."""

    @abstractmethod
    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        """Restore the original shape, dtype and device."""

    def fake_quantize(
        self, x: torch.Tensor, plan: Optional[PrecisionPlan] = None
    ) -> torch.Tensor:
        return self.dequantize(self.quantize(x, plan))

    def calibrate(self, samples: Any) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} has no calibration implementation"
        )

    @classmethod
    def create_from_config(cls, config: KVQuantizerConfig) -> "KVQuantizer":
        return cast(KVQuantizer, KVQuantizerRegistry.create(config))


class ScalarQuantizer(KVQuantizer):
    """Fixed numerical-format family in the diagram (also scalar codebooks exist)."""


class CodebookQuantizer(KVQuantizer):
    """TODO: fitted/analytic codebooks, calibration artifacts and decoding."""


class PendingQuantizer(KVQuantizer):
    def __init__(self, config: KVQuantizerConfig) -> None:
        super().__init__(config)
        raise NotImplementedError(
            f"{type(self).__name__} is a scaffold; implement before selecting it"
        )

    def quantize(
        self, x: torch.Tensor, plan: Optional[PrecisionPlan] = None
    ) -> QuantizedTensor:
        raise NotImplementedError

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        raise NotImplementedError
