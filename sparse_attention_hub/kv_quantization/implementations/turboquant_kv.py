"""TurboQuant MSE/prod KV recipes using the shared immediate-reconstruction flow."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from ..base import KVQuantization, KVQuantizationConfig
from ..policies.cache.implementations.immediate import FullQuantizationCachePolicyConfig
from ..policies.precision.implementations.fixed import FixedPrecisionConfig
from ..quantizers.implementations.turboquant import TurboQuantQuantizerConfig
from ..registry import KVQuantizationRegistry
from ..transforms.implementations.identity import IdentityTransformConfig


@dataclass(frozen=True)
class TurboQuantKVConfig(KVQuantizationConfig):
    """Uniform whole-head profile: 3-bit MSE K/V, no exact residual regions.

    Select variant="prod" in a K/V codec to spend one of its nominal bits on
    QJL correction. Fractional-bit outlier benchmarks are a separate profile.
    Normalization and rotation live in the codec; outer transforms are identity.
    """

    key_transform: IdentityTransformConfig = field(
        default_factory=IdentityTransformConfig
    )
    value_transform: IdentityTransformConfig = field(
        default_factory=IdentityTransformConfig
    )
    key_quantizer: TurboQuantQuantizerConfig = field(
        default_factory=TurboQuantQuantizerConfig
    )
    value_quantizer: TurboQuantQuantizerConfig = field(
        default_factory=TurboQuantQuantizerConfig
    )
    cache_policy: FullQuantizationCachePolicyConfig = field(
        default_factory=FullQuantizationCachePolicyConfig
    )
    store_codes: bool = True
    quantize_prefill: bool = True


def validate_turboquant_config(config: TurboQuantKVConfig) -> None:
    """Keep the named profile's codec, coordinate and retention contracts explicit."""
    for kind in ("key", "value"):
        if type(getattr(config, f"{kind}_transform")) is not IdentityTransformConfig:
            raise ValueError(
                "TurboQuant owns its rotation; outer transforms must be identity"
            )
        if type(getattr(config, f"{kind}_quantizer")) is not TurboQuantQuantizerConfig:
            raise ValueError("TurboQuant requires TurboQuantQuantizerConfig")
    if type(config.cache_policy) is not FullQuantizationCachePolicyConfig:
        raise ValueError("TurboQuant requires FullQuantizationCachePolicyConfig")
    if type(config.precision_policy) is not FixedPrecisionConfig:
        raise ValueError("TurboQuant requires FixedPrecisionConfig")
    if (
        type(config.store_codes) is not bool
        or type(config.quantize_prefill) is not bool
    ):
        raise ValueError("TurboQuant storage/prefill flags must be boolean")


def validate_turboquant_layout(keys: torch.Tensor, values: torch.Tensor) -> None:
    """Validate the head schema before the shared helper creates request state."""
    if keys.ndim != 4 or values.ndim != 4 or keys.shape[:3] != values.shape[:3]:
        raise ValueError(
            "TurboQuant requires matching [batch, kv_heads, tokens, channels] K/V"
        )
    for tensor in (keys, values):
        if tensor.shape[-1] < 2 or tensor.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError(
                "TurboQuant requires FP16/BF16/FP32 heads of dimension at least 2"
            )


@KVQuantizationRegistry.register(TurboQuantKVConfig)
class TurboQuantKV(KVQuantization):
    """Reconstruct previous and arriving K/V before ordinary attention."""

    config: TurboQuantKVConfig

    def __init__(self, config: TurboQuantKVConfig) -> None:
        validate_turboquant_config(config)
        super().__init__(config)

    def process(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        meta_data: Dict[str, Any],
        *,
        num_new_tokens: int,
        phase: str = "decode",
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        validate_turboquant_layout(keys, values)
        return self._process_kv_history(
            layer_idx,
            keys,
            values,
            meta_data,
            num_new_tokens=num_new_tokens,
            phase=phase,
            position_ids=position_ids,
            reconstruct_current=True,
        )
