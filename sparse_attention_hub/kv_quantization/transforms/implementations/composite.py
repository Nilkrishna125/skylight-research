"""CompositeTransform component and its configuration."""

from dataclasses import dataclass
from typing import Any, Tuple

import torch

from ...context import AttentionContext
from ...registry import KVTransformRegistry
from ..base import KVTransform, KVTransformConfig


@dataclass(frozen=True)
class CompositeTransformConfig(KVTransformConfig):
    transforms: Tuple[KVTransformConfig, ...] = ()


@KVTransformRegistry.register(CompositeTransformConfig)
class CompositeTransform(KVTransform):
    def __init__(self, config: CompositeTransformConfig) -> None:
        super().__init__(config)
        self.transforms = [KVTransform.create_from_config(c) for c in config.transforms]

    def forward(
        self, x: torch.Tensor, context: AttentionContext
    ) -> Tuple[torch.Tensor, Any]:
        states = []
        for transform in self.transforms:
            x, state = transform.forward(x, context)
            states.append(state)
        return x, states

    def inverse(
        self, x: torch.Tensor, context: AttentionContext, state: Any
    ) -> torch.Tensor:
        for transform, saved in reversed(list(zip(self.transforms, state))):
            x = transform.inverse(x, context, saved)
        return x
