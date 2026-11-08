"""QuaRot KV accuracy profile with original-coordinate reconstruction."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from ..base import KVQuantization, KVQuantizationConfig
from ..policies.cache.implementations.immediate import FullQuantizationCachePolicyConfig
from ..policies.precision.implementations.fixed import FixedPrecisionConfig
from ..quantizers.implementations.quarot import QuaRotQuantizerConfig
from ..registry import KVQuantizationRegistry
from ..transforms.implementations.hadamard import HadamardTransformConfig


@dataclass(frozen=True)
class QuaRotKVConfig(KVQuantizationConfig):
    """Rotated per-token KV groups; no exact residual, including current tokens.

    Defaults: asymmetric INT4, group size 128, clipping 0.95, quantized prefill
    and packed storage. Weights and ordinary model activations remain untouched.
    """

    key_transform: HadamardTransformConfig = field(
        default_factory=lambda: HadamardTransformConfig(preserve_input_dtype=True)
    )
    value_transform: HadamardTransformConfig = field(
        default_factory=lambda: HadamardTransformConfig(preserve_input_dtype=True)
    )
    key_quantizer: QuaRotQuantizerConfig = field(
        default_factory=lambda: QuaRotQuantizerConfig(parameter_mode="key_head")
    )
    value_quantizer: QuaRotQuantizerConfig = field(
        default_factory=lambda: QuaRotQuantizerConfig(parameter_mode="value_group")
    )
    cache_policy: FullQuantizationCachePolicyConfig = field(
        default_factory=FullQuantizationCachePolicyConfig
    )
    store_codes: bool = True
    quantize_prefill: bool = True


def validate_quarot_config(config: QuaRotKVConfig) -> None:
    """Reject component substitutions that change the named method's contract."""
    for kind in ("key", "value"):
        transform = getattr(config, f"{kind}_transform")
        quantizer = getattr(config, f"{kind}_quantizer")
        if (
            type(transform) is not HadamardTransformConfig
            or not transform.preserve_input_dtype
        ):
            raise ValueError(
                "QuaRot requires HadamardTransform with preserve_input_dtype=True"
            )
        if type(quantizer) is not QuaRotQuantizerConfig:
            raise ValueError("QuaRot requires QuaRotQuantizerConfig")
        expected = "key_head" if kind == "key" else "value_group"
        if quantizer.parameter_mode != expected:
            raise ValueError(f"QuaRot {kind} requires parameter_mode={expected}")
    if type(config.cache_policy) is not FullQuantizationCachePolicyConfig:
        raise ValueError("QuaRot requires FullQuantizationCachePolicyConfig")
    if type(config.precision_policy) is not FixedPrecisionConfig:
        raise ValueError("QuaRot requires FixedPrecisionConfig")
    if (
        type(config.quantize_prefill) is not bool
        or type(config.store_codes) is not bool
    ):
        raise ValueError("QuaRot prefill/storage flags must be boolean")


def validate_quarot_layout(
    keys: torch.Tensor, values: torch.Tensor, config: QuaRotKVConfig
) -> None:
    """Validate layout before the shared coordinator creates or mutates state."""
    if keys.ndim != 4 or values.ndim != 4 or keys.shape[:3] != values.shape[:3]:
        raise ValueError(
            "QuaRot requires matching [batch, kv_heads, tokens, channels] K/V"
        )
    for kind, tensor, quantizer in (
        ("key", keys, config.key_quantizer),
        ("value", values, config.value_quantizer),
    ):
        dimension = tensor.shape[-1]
        if dimension < 1 or dimension & (dimension - 1):
            raise ValueError("QuaRot requires a power-of-two head dimension")
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("QuaRot K/V must be FP16, BF16 or FP32")
        if dimension % quantizer.group_size or (
            kind == "key" and dimension != quantizer.group_size
        ):
            raise ValueError(
                "QuaRot needs a complete key head group and complete value groups"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError("QuaRot cannot quantize NaN or infinity")


@KVQuantizationRegistry.register(QuaRotKVConfig)
class QuaRotKV(KVQuantization):
    """Quantize arriving tokens, reconstruct K/V, then run ordinary attention."""

    config: QuaRotKVConfig

    def __init__(self, config: QuaRotKVConfig) -> None:
        validate_quarot_config(config)
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
        validate_quarot_layout(keys, values, self.config)
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
