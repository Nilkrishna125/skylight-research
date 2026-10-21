"""Own and reconstruct request-local KV histories."""

from typing import Any, Callable, Dict, List, Tuple

import torch

from ..policies.cache.base import CachePolicy
from .block import StoredChunk
from .state import LayerKVState, TensorCacheState

CACHE_KEY = "kv_quantization_store"
PENDING_LOCAL = "pending_local"

# encode(kind, region, tensor, start_token) -> StoredChunk; region is sink/middle/local.
Encoder = Callable[[str, str, torch.Tensor, int], StoredChunk]
# decode(kind, chunk) -> tensor in original coordinates, dtype and device.
Decoder = Callable[[str, StoredChunk], torch.Tensor]


class QuantizedKVStore:
    def __init__(
        self, owner: object, *, store_codes: bool, sink_tokens: Dict[str, int]
    ) -> None:
        self.owner = owner
        self.store_codes = store_codes
        self.sink_tokens = sink_tokens
        self.layers: Dict[int, LayerKVState] = {}

    def seen_tokens(self, layer_idx: int) -> int:
        state = self.layers.get(layer_idx)
        return 0 if state is None else state.seen_tokens

    @staticmethod
    def _schema(x: torch.Tensor) -> Tuple[Any, ...]:
        return x.shape[0], x.shape[1], x.shape[3], x.dtype, x.device

    def append(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        encode: Encoder,
    ) -> int:
        """Inputs are full HF history. Own only the unseen suffix.

        The first configured tokens go to the sink (encoded now, since sinks never
        move). The rest join the local window as pending full-precision chunks;
        ``compress_ready`` moves old ones to the middle and encodes the remainder.
        """
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
            target: TensorCacheState = getattr(state, kind)
            new = full[..., state.seen_tokens :, :].detach()
            sink_count = min(
                new_count, max(0, self.sink_tokens[kind] - target.sink_tokens)
            )
            if sink_count:
                target.sink.append(
                    encode(kind, "sink", new[..., :sink_count, :], state.seen_tokens)
                )
            rest = new[..., sink_count:, :]
            if rest.shape[-2]:
                target.recent.append(
                    StoredChunk(
                        state.seen_tokens + sink_count,
                        int(rest.shape[-2]),
                        region=PENDING_LOCAL,
                        full_precision=rest.clone(),
                    )
                )
        state.seen_tokens = int(keys.shape[-2])
        return new_count

    def compress_ready(
        self,
        layer_idx: int,
        policy: CachePolicy,
        encode: Encoder,
        decode: Decoder,
        *,
        phase: str = "before_attention",
    ) -> None:
        state = self.layers[layer_idx]
        for kind in ("key", "value"):
            target: TensorCacheState = getattr(state, kind)
            while True:
                count = policy.next_chunk_tokens(kind, target, phase=phase)
                if not count:
                    break
                if count < 0 or count > target.recent_tokens:
                    raise ValueError("Cache policy returned an invalid chunk length")
                start = target.recent[0].start_token
                moved = self._take_local(kind, target, count, encode, decode)
                target.chunks.append(encode(kind, "middle", moved, start))
            # Tokens still in the local window are stored with the local quantizer.
            target.recent = [
                encode(kind, "local", chunk.full_precision, chunk.start_token)
                if chunk.region == PENDING_LOCAL and chunk.full_precision is not None
                else chunk
                for chunk in target.recent
            ]

    @staticmethod
    def _take_local(
        kind: str,
        target: TensorCacheState,
        count: int,
        encode: Encoder,
        decode: Decoder,
    ) -> torch.Tensor:
        """Remove the oldest ``count`` local tokens and return their values.

        Quantized local tokens are returned as decoded from their stored local copy,
        so moving them to the middle re-quantizes that copy (not the original).
        """
        taken: List[torch.Tensor] = []
        remaining = count
        while remaining:
            chunk = target.recent[0]
            values = decode(kind, chunk)
            if chunk.token_count <= remaining:
                taken.append(values)
                target.recent.pop(0)
                remaining -= chunk.token_count
                continue
            taken.append(values[..., :remaining, :])
            rest = values[..., remaining:, :].clone()
            rest_start = chunk.start_token + remaining
            if chunk.is_encoded:
                target.recent[0] = encode(kind, "local", rest, rest_start)
            else:
                target.recent[0] = StoredChunk(
                    rest_start,
                    int(rest.shape[-2]),
                    region=chunk.region,
                    full_precision=rest,
                )
            remaining = 0
        return torch.cat(taken, dim=-2)

    def get(
        self, layer_idx: int, decode: Decoder
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct temporary K/V, preserving original order and KV-head layout."""
        outputs = []
        layer = self.layers[layer_idx]
        for kind in ("key", "value"):
            state: TensorCacheState = getattr(layer, kind)
            parts = [decode(kind, chunk) for chunk in state.all_chunks()]
            if not parts:
                raise ValueError("Cannot read an empty KV history")
            full = torch.cat(parts, dim=-2)
            if full.shape[-2] != layer.seen_tokens:
                raise RuntimeError(
                    "Stored KV regions do not cover the complete token history"
                )
            outputs.append(full)
        return outputs[0], outputs[1]

    @staticmethod
    def _chunk_bytes(chunk: StoredChunk) -> int:
        for tensor in (chunk.full_precision, chunk.dequantized):
            if tensor is not None:
                return tensor.numel() * tensor.element_size()
        return 0 if chunk.quantized is None else chunk.quantized.storage_bytes()

    def stats(self) -> Dict[str, Any]:
        owned_bytes = 0
        layers = {}
        for index, layer in self.layers.items():
            info = {"seen_tokens": layer.seen_tokens}
            for kind in ("key", "value"):
                state: TensorCacheState = getattr(layer, kind)
                info[f"{kind}_sink_tokens"] = state.sink_tokens
                info[f"{kind}_compressed_tokens"] = state.compressed_tokens
                info[f"{kind}_recent_tokens"] = state.recent_tokens
                info[f"{kind}_encoded_tokens"] = sum(
                    end - start for start, end in state.encoded_ranges()
                )
                owned_bytes += sum(
                    self._chunk_bytes(chunk) for chunk in state.all_chunks()
                )
            layers[index] = info
        return {"store_bytes": owned_bytes, "layers": layers}

    def clear(self) -> None:
        self.layers.clear()
