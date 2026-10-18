"""Own and reconstruct request-local KV histories."""

from typing import Any, Callable, Dict, Tuple

import torch

from ..policies.cache.base import CachePolicy
from .block import StoredChunk
from .state import LayerKVState

CACHE_KEY = "kv_quantization_store"


class QuantizedKVStore:
    def __init__(
        self, owner: object, *, store_codes: bool, num_sink_tokens: int
    ) -> None:
        self.owner = owner
        self.store_codes = store_codes
        self.num_sink_tokens = num_sink_tokens
        self.layers: Dict[int, LayerKVState] = {}

    def seen_tokens(self, layer_idx: int) -> int:
        state = self.layers.get(layer_idx)
        return 0 if state is None else state.seen_tokens

    @staticmethod
    def _schema(x: torch.Tensor) -> Tuple[Any, ...]:
        return x.shape[0], x.shape[1], x.shape[3], x.dtype, x.device

    def append(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> int:
        """Inputs are full HF history. Own only the unseen suffix."""
        if keys.ndim != 4 or values.ndim != 4 or keys.shape[:3] != values.shape[:3]:
            raise ValueError(
                "Expected matching [batch, kv_heads, tokens, channels] K/V"
            )
        if not keys.is_floating_point() or not values.is_floating_point():
            raise ValueError("Native K/V inputs must be floating point")
        state = self.layers.setdefault(layer_idx, LayerKVState())
        if keys.shape[-2] < state.seen_tokens:
            raise ValueError(
                "KV history shrank; eviction/rollback requires a different cache policy"
            )
        schemas = (self._schema(keys), self._schema(values))
        if state.key_schema is not None and schemas != (
            state.key_schema,
            state.value_schema,
        ):
            raise ValueError(
                "Batch/head/channel shape, dtype or device changed; start a fresh store"
            )
        state.key_schema, state.value_schema = schemas
        new_count = int(keys.shape[-2]) - state.seen_tokens
        if new_count == 0:
            return 0
        for kind, full in (("key", keys), ("value", values)):
            target = getattr(state, kind)
            new = full[..., state.seen_tokens :, :].detach().clone()
            sink_count = min(
                new_count, max(0, self.num_sink_tokens - target.sink_tokens)
            )
            if sink_count:
                sink = new[..., :sink_count, :].clone()
                target.sink = (
                    sink if target.sink is None else torch.cat([target.sink, sink], -2)
                )
            new = new[..., sink_count:, :].clone()
            if new.shape[-2]:
                target.recent = (
                    new
                    if target.recent is None
                    else torch.cat([target.recent, new], -2)
                )
        state.seen_tokens = int(keys.shape[-2])
        return new_count

    def compress_ready(
        self,
        layer_idx: int,
        policy: CachePolicy,
        encode: Callable[[str, torch.Tensor, int], StoredChunk],
        *,
        phase: str = "before_attention",
    ) -> None:
        state = self.layers[layer_idx]
        for kind in ("key", "value"):
            target = getattr(state, kind)
            while True:
                count = policy.next_chunk_tokens(kind, target, phase=phase)
                if not count:
                    break
                if count < 0 or count > target.recent_tokens:
                    raise ValueError("Cache policy returned an invalid chunk length")
                assert target.recent is not None
                start = target.sink_tokens + target.compressed_tokens
                chunk: StoredChunk = encode(kind, target.recent[..., :count, :], start)
                target.chunks.append(chunk)
                target.recent = target.recent[..., count:, :].clone()

    def get(
        self, layer_idx: int, decode: Callable[[str, StoredChunk], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct temporary K/V, preserving original order and KV-head layout."""
        outputs = []
        layer = self.layers[layer_idx]
        for kind in ("key", "value"):
            state = getattr(layer, kind)
            parts = [] if state.sink is None else [state.sink]
            parts.extend(decode(kind, chunk) for chunk in state.chunks)
            if state.recent is not None and state.recent.numel():
                parts.append(state.recent)
            if not parts:
                raise ValueError("Cannot read an empty KV history")
            full = torch.cat(parts, dim=-2)
            if full.shape[-2] != layer.seen_tokens:
                raise RuntimeError(
                    "Stored KV regions do not cover the complete token history"
                )
            outputs.append(full)
        return outputs[0], outputs[1]

    def stats(self) -> Dict[str, Any]:
        owned_bytes = 0
        layers = {}
        for index, layer in self.layers.items():
            info = {"seen_tokens": layer.seen_tokens}
            for kind in ("key", "value"):
                state = getattr(layer, kind)
                info[f"{kind}_compressed_tokens"] = state.compressed_tokens
                info[f"{kind}_recent_tokens"] = state.recent_tokens
                info[f"{kind}_sink_tokens"] = state.sink_tokens
                for tensor in (state.sink, state.recent):
                    if tensor is not None:
                        owned_bytes += tensor.numel() * tensor.element_size()
                for chunk in state.chunks:
                    if chunk.quantized is not None:
                        owned_bytes += chunk.quantized.storage_bytes()
                    if chunk.dequantized is not None:
                        owned_bytes += (
                            chunk.dequantized.numel() * chunk.dequantized.element_size()
                        )
            layers[index] = info
        return {"store_bytes": owned_bytes, "layers": layers}

    def clear(self) -> None:
        self.layers.clear()
