"""KV quantization coordinator and its configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, cast

import torch

from sparse_attention_hub.metric_logging.logger import MicroMetricLogger

from .cache.block import StoredChunk
from .cache.store import CACHE_KEY, QuantizedKVStore
from .context import AttentionContext, record_trace
from .metrics.quant_error import QuantErrorMetric
from .policies.cache.base import CachePolicy, CachePolicyConfig
from .policies.cache.implementations.residual_policy import ResidualCachePolicyConfig
from .policies.precision.base import PrecisionPolicy, PrecisionPolicyConfig
from .policies.precision.implementations.fixed import FixedPrecisionConfig
from .quantizers.base import KVQuantizer, KVQuantizerConfig
from .quantizers.implementations.integer import IntQuantizerConfig
from .registry import (
    CachePolicyRegistry,
    KVQuantizationRegistry,
    PrecisionPolicyRegistry,
)
from .transforms.base import KVTransform, KVTransformConfig
from .transforms.implementations.identity import IdentityTransformConfig


@dataclass(frozen=True)
class KVQuantizationConfig:
    key_transform: KVTransformConfig = field(default_factory=IdentityTransformConfig)
    value_transform: KVTransformConfig = field(default_factory=IdentityTransformConfig)
    key_quantizer: KVQuantizerConfig = field(default_factory=IntQuantizerConfig)
    value_quantizer: KVQuantizerConfig = field(default_factory=IntQuantizerConfig)
    precision_policy: PrecisionPolicyConfig = field(
        default_factory=FixedPrecisionConfig
    )
    cache_policy: CachePolicyConfig = field(default_factory=ResidualCachePolicyConfig)
    store_codes: bool = False
    error_metric: bool = False
    quantize_prefill: bool = False


@KVQuantizationRegistry.register(KVQuantizationConfig)
class KVQuantization:
    def __init__(self, config: KVQuantizationConfig) -> None:
        self.config = config
        # Construction fails immediately for unimplemented components.
        self.key_transform = KVTransform.create_from_config(config.key_transform)
        self.value_transform = KVTransform.create_from_config(config.value_transform)
        self.key_quantizer = KVQuantizer.create_from_config(config.key_quantizer)
        self.value_quantizer = KVQuantizer.create_from_config(config.value_quantizer)
        self.precision_policy = cast(
            PrecisionPolicy, PrecisionPolicyRegistry.create(config.precision_policy)
        )
        self.cache_policy = cast(
            CachePolicy, CachePolicyRegistry.create(config.cache_policy)
        )
        self.error_metric = QuantErrorMetric() if config.error_metric else None
        self._owner = object()

    @classmethod
    def create_from_config(cls, config: KVQuantizationConfig) -> "KVQuantization":
        return cast(KVQuantization, KVQuantizationRegistry.create(config))

    def process(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        meta_data: Dict[str, Any],
        *,
        phase: str = "decode",
        position_ids: Optional[torch.Tensor] = None,
        rope_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            not isinstance(layer_idx, int)
            or isinstance(layer_idx, bool)
            or layer_idx < 0
        ):
            raise ValueError(
                "KV quantization requires a non-negative integer layer_idx"
            )
        if phase not in ("prefill", "decode"):
            raise ValueError("kv_quantization_phase must be 'prefill' or 'decode'")
        if phase == "prefill" and not self.config.quantize_prefill:
            if CACHE_KEY in meta_data:
                raise ValueError("Start a fresh request before prefilling new history")
            record_trace(
                meta_data, "KVQuantization.process: native prefill", layer=layer_idx
            )
            return keys, values

        store = meta_data.get(CACHE_KEY)
        if store is None:
            store = QuantizedKVStore(
                self._owner,
                store_codes=self.config.store_codes,
                num_sink_tokens=self.config.cache_policy.num_sink_tokens,
            )
            meta_data[CACHE_KEY] = store
            record_trace(meta_data, "QuantizedKVStore: create")
        if not isinstance(store, QuantizedKVStore) or store.owner is not self._owner:
            raise ValueError(
                "Metadata belongs to another quantization configuration; use a fresh request"
            )

        imported = store.append(layer_idx, keys, values)
        record_trace(
            meta_data,
            "QuantizedKVStore.append: unseen suffix",
            layer=layer_idx,
            tokens=imported,
        )

        self._compress(
            store,
            layer_idx,
            meta_data,
            "before_attention",
            phase,
            position_ids,
            rope_context,
        )
        reconstructed_keys, reconstructed_values = store.get(
            layer_idx, self._decode_chunk
        )
        record_trace(
            meta_data,
            "QuantizedKVStore.get: decode + inverse + FP regions",
            layer=layer_idx,
        )
        if self.error_metric is not None:
            state = store.layers[layer_idx]
            self.error_metric.log("key", keys, reconstructed_keys, state.key, layer_idx)
            self.error_metric.log(
                "value", values, reconstructed_values, state.value, layer_idx
            )
            MicroMetricLogger().log(
                "kv_quantization_store_bytes",
                store.stats()["store_bytes"],
                metadata={"layer_idx": layer_idx},
            )
        return reconstructed_keys, reconstructed_values

    def _encode_chunk(
        self,
        kind: str,
        tensor: torch.Tensor,
        start: int,
        layer_idx: int,
        metadata: Dict[str, Any],
        phase: str,
        position_ids: Optional[torch.Tensor],
        rope_context: Optional[Dict[str, Any]],
    ) -> StoredChunk:
        context = AttentionContext(
            layer_idx, kind, start, phase, position_ids, rope_context
        )
        transform = self.key_transform if kind == "key" else self.value_transform
        quantizer = self.key_quantizer if kind == "key" else self.value_quantizer
        record_trace(
            metadata, "KVTransform.forward", layer=layer_idx, kind=kind, start=start
        )
        transformed, saved = transform.forward(tensor, context)
        positions = torch.arange(start, start + tensor.shape[-2], device=tensor.device)
        plan = self.precision_policy.bits_for(
            layer_idx,
            transformed,
            positions,
            tensor_kind=kind,
            default_bits=quantizer.config.bits,
        )
        record_trace(
            metadata,
            "PrecisionPolicy.bits_for -> KVQuantizer.quantize",
            layer=layer_idx,
            kind=kind,
            bits=plan.bits,
        )
        encoded = quantizer.quantize(transformed, plan)
        chunk = StoredChunk(start, tensor.shape[-2], context, transform_state=saved)
        if self.config.store_codes:
            chunk.quantized = encoded
        else:
            chunk.dequantized = (
                transform.inverse(
                    quantizer.dequantize(encoded),
                    context,
                    saved,
                )
                .detach()
                .clone()
            )
        return chunk

    def _decode_chunk(self, kind: str, chunk: StoredChunk) -> torch.Tensor:
        if chunk.dequantized is not None:
            return chunk.dequantized
        quantizer = self.key_quantizer if kind == "key" else self.value_quantizer
        transform = self.key_transform if kind == "key" else self.value_transform
        assert chunk.quantized is not None
        return transform.inverse(
            quantizer.dequantize(chunk.quantized),
            chunk.context,
            chunk.transform_state,
        )

    def _compress(
        self,
        store: QuantizedKVStore,
        layer_idx: int,
        metadata: Dict[str, Any],
        timing: str,
        phase: str,
        position_ids: Optional[torch.Tensor],
        rope_context: Optional[Dict[str, Any]],
    ) -> None:
        def encode(kind: str, tensor: torch.Tensor, start: int) -> StoredChunk:
            return self._encode_chunk(
                kind,
                tensor,
                start,
                layer_idx,
                metadata,
                phase,
                position_ids,
                rope_context,
            )

        store.compress_ready(layer_idx, self.cache_policy, encode, phase=timing)

    def after_attention(
        self,
        layer_idx: int,
        meta_data: Dict[str, Any],
        *,
        phase: str = "decode",
        position_ids: Optional[torch.Tensor] = None,
        rope_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        store = meta_data.get(CACHE_KEY)
        if store is not None:
            self._compress(
                store,
                layer_idx,
                meta_data,
                "after_attention",
                phase,
                position_ids,
                rope_context,
            )
            self.cache_policy.after_attention(store, layer_idx)

    def clear_request(self, meta_data: Dict[str, Any]) -> None:
        store = meta_data.pop(CACHE_KEY, None)
        if store is not None:
            store.clear()

    def calibrate(self, kv_samples: Any) -> None:
        """TODO: dispatch method-specific calibration and persist its artifacts."""
        raise NotImplementedError("Calibration recipes are scaffolded, not implemented")


class PendingRecipe(KVQuantization):
    def __init__(self, config: KVQuantizationConfig) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} is a recipe scaffold. Implement and validate its "
            "numerics, calibration, storage and cache policy before selecting it. "
            "Use KVQuantizationConfig with IntQuantizerConfig for the working baseline."
        )
