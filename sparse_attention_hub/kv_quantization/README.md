# KV quantization implementation scaffold

Run from the repository root with the project's Python environment (Python 3.10+):

```bash
python tutorials/py_examples/06_kv_quantization_flow.py --trace
```

This runs a tiny randomly initialized, two-layer Hugging Face Llama on CPU.
It downloads no models or data. OFF, FAKE and PACKED use identical weights and
fixed token IDs; the script checks exact fake/packed logit parity, both layers'
incremental state, the unchanged native HF cache, and cleanup. These are plumbing
and numerical checks, not a language-quality benchmark. Use `--mode packed`,
`--mode fake`, `--mode off`, or `--bits 2` to explore individual paths.

The actual call flow is:

```text
HF model: RoPE -> native cache.update -> attention registry callback
  adapters/huggingface.py: custom_attention_callable (adds layer_idx, sparse_meta_data)
    research_attention/base.py: ResearchAttention.custom_attention
      sparse_attention/base.py: SparseAttention.quantize_kv (returns K, V unchanged when off)
        kv_quantization/base.py: KVQuantization.process
          cache/store.py: append only unseen tokens, independently for each layer
          policies/cache/: decide which K/V tokens leave the local window
          transforms/: forward transform
          policies/precision/: precision plan
          quantizers/: encode + pack, with per-group parameters
          cache/store.py: read sink + middle + local regions, decoding quantized ones
          quantizers/ + transforms/: decode and inverse transform
      maskers + attention on the reconstructed K, V (every layer, before layer gating)
  adapters/huggingface.py: per-question finally -> SparseAttention.clear_request_state
```

## Files and extension points

Each component family follows the repository's `base.py` / `implementations/`
pattern. An implementation and its configuration live in the same file.
Package `__init__.py` files are omitted from this tree:

```text
kv_quantization/
├── base.py                         # KVQuantizationConfig, KVQuantization
├── registry.py
├── context.py                      # AttentionContext, tracing helper
├── implementations/                # Complete methods and their configs
│   ├── kivi.py
│   ├── quarot_kv.py
│   ├── turboquant_kv.py
│   ├── kvquant_nuq.py
│   ├── kitty.py
│   └── nvfp4_kv.py
├── transforms/
│   ├── base.py
│   └── implementations/
│       ├── identity.py
│       ├── hadamard.py
│       ├── random_hadamard.py
│       ├── random_rotation.py
│       └── learned_rotation.py   # stub
├── quantizers/
│   ├── base.py
│   └── implementations/
│       ├── integer.py
│       ├── fp8.py
│       ├── nvfp4.py
│       ├── fake_nvfp4.py
│       ├── mxfp.py
│       └── non_uniform.py
├── policies/
│   ├── precision/
│   │   ├── base.py
│   │   └── implementations/
│   │       ├── fixed.py
│   │       ├── channel_boost.py
│   │       └── outlier_token.py
│   └── cache/
│       ├── base.py
│       └── implementations/
│           ├── default.py          # DefaultCachePolicy
│           ├── sink_and_local.py   # SinkAndLocalCachePolicy, KittyCachePolicy
│           ├── immediate.py        # SinkOnlyCachePolicy, FullQuantizationCachePolicy
│           └── local_only.py       # LocalOnlyCachePolicy
├── cache/
│   ├── store.py                    # QuantizedKVStore
│   ├── state.py                    # LayerKVState, TensorCacheState
│   └── block.py                    # QuantizedTensor, StoredChunk
├── metrics/
│   └── quant_error.py
└── README.md
```

| Location | Main classes / responsibility |
|---|---|
| `base.py` | `KVQuantizationConfig` and `KVQuantization`: select and orchestrate components |
| `registry.py` | Independent config-type factories; implementations use their registration decorators |
| `context.py` | `AttentionContext` and opt-in tracing; metadata only, no KV storage |
| `transforms/` | `KVTransform`; working Identity, Hadamard, RandomHadamard and RandomRotation; LearnedRotation scaffolded |
| `quantizers/` | `KVQuantizer`, `QuantGranularity`, working grouped `IntQuantizer`; FP/codebook scaffolds |
| `policies/precision/` | `PrecisionPolicy`, `PrecisionPlan`, working fixed bits; mixed-bit scaffolds |
| `policies/cache/` | `CachePolicy`, `RegionConfig` (sink/local size and quantizer), and six working policies (Default, SinkAndLocal, Kitty, SinkOnly, FullQuantization, LocalOnly) |
| `cache/` | `QuantizedKVStore`, per-layer state, encoded tensor records and chunks |
| `metrics/quant_error.py` | Optional reconstruction errors on quantized regions; stored tensor bytes |
| `implementations/` | KIVI, QuaRotKV, TurboQuantKV, KVQuantNUQ, Kitty and NVFP4KV method declarations |

For example, `implementations/kivi.py` assembles the complete method from
components, including `SinkAndLocalCachePolicyConfig` for its buffering. Methods share
components instead of defining new ones each time.

Base configurations live beside their contracts in each family's `base.py`.
Concrete configurations live beside their implementation: `IntQuantizerConfig`
and `IntQuantizer` are both in `quantizers/implementations/integer.py`.
There is no central file containing every method's configuration.

Existing package-level imports remain available:

```python
from sparse_attention_hub.kv_quantization import (
    IntQuantizerConfig,
    KVQuantizationConfig,
    QuantizedKVStore,
)
```

To add a component, put its class and config in a new implementation file, apply
the appropriate registry decorator, and import/export it in the package-level
`__init__.py` so registration runs. Shared base classes and other implementations
normally do not need changes. Unit tests are grouped under `quantizers/`,
`cache/`, `policies/cache/`, and `implementations/`, with coordinator and attention
wrapper tests at the unit package root.

`SparseAttentionConfig.kv_quantization_config` turns it on. `SparseAttention` builds
the coordinator once and exposes `quantize_kv`, which each attention subclass calls
first in its `custom_attention` (`ResearchAttention` does; efficient attention is not
run by the adapter yet). `after_attention` is an extension point that nothing calls
yet. The config field is keyword-only to preserve required subclass fields, requiring
Python 3.10 or newer. Existing config calls without quantization keep the OFF path.

## What is implemented

Grouped uniform INT2/4/8 quantization computes FP32 statistics and stores a
floating offset (the group minimum) and a scale per group. Asymmetric encoding is
`round((x - offset) / scale)`, with `scale = (max - offset) / (2**bits - 1)`. The
optional symmetric mode uses signed `[-qmax, qmax]`, stored as unsigned codes, and
stores no offset. Codes are packed into uint8: four INT2 or two INT4 entries per byte.
This is a reference PyTorch implementation, not an optimized GPU kernel.

Scales and offsets are stored in `KVQuantizerConfig.scale_dtype` / `offset_dtype`
(`ParamDType.FP32`, `FP16` or `BF16`; default FP16 for both, set separately for keys
and values). They are rounded to that dtype *before* the codes are computed, so
decoding uses exactly the stored values. Rounding is conservative: offsets round down
and scales round up by at most one step of the dtype, so the stored range always covers
the group and nothing is clipped; a coarser dtype only widens the range. bf16 has few
mantissa bits, so offsets far from zero cost more precision than with fp16. FP16 halves
the parameter overhead of FP32: INT2 with groups of 32 costs 2 + 32/32 = 3 bits per
value instead of 4. A parameter beyond the dtype's range raises a clear error.

`PER_TOKEN` groups channels; `PER_CHANNEL` groups tokens. `PER_BLOCK` groups
consecutive channels within one token, and `group_size` must divide head_dim, so a
block never spans two tokens.
Incomplete groups repeat the last entry for statistics and discard padding on
decode. For paper reproduction, cache-policy boundaries must match that paper's
grouping instead of relying on this baseline padding convention.

Each history (keys, and separately values) has three regions:

```text
[ sink: first tokens ][ middle: older history ][ local: newest tokens ]
```

The cache policy sets the sink and local regions independently for keys and values.
Each is a `RegionConfig(tokens, quantizer)`, and `quantizer=None` keeps it in its
original precision. The middle region uses `KVQuantizationConfig.key_quantizer` /
`value_quantizer`.

```python
DefaultCachePolicyConfig(
    key_sink=RegionConfig(4, IntQuantizerConfig(bits=8)),    # first 4 key tokens in INT8
    value_sink=RegionConfig(4),                              # first 4 value tokens in FP
    key_local=RegionConfig(128, IntQuantizerConfig(bits=4)), # newest 128 key tokens in INT4
    value_local=RegionConfig(64),                            # newest 64 value tokens in FP
)
```

Every policy shares those region settings and differs only in *when* local tokens
move to the middle:

| Policy (config) | Rule | Full-precision tail |
|---|---|---|
| `DefaultCachePolicy` (default) | move the oldest `*_block_tokens` once local holds `local.tokens + block` | `local` to `local + block - 1` |
| `SinkAndLocalCachePolicy` (KIVI's buffering) | keys: buffer `key_local.tokens` (R), then move whole multiples of R; values: exact rolling window | keys 0 to R-1, values exactly R |
| `KittyCachePolicy` | the sink-and-local rule with 32 sinks by default (keys page out 128 at a time) | as KIVI, after the sinks |
| `SinkOnlyCachePolicy` | sinks only; every other token is quantized on arrival | sinks only |
| `FullQuantizationCachePolicy` | no sinks, no local window; everything is quantized on arrival | none |
| `LocalOnlyCachePolicy` | no sinks; an exact local window (blocks of 1 by default) | exactly `local.tokens` |

To add a policy, subclass `CachePolicyConfig` (for any new settings) and `CachePolicy`,
implement `next_chunk_tokens` and `min_move_tokens`, and register it with
`@CachePolicyRegistry.register(...)`. A per-channel middle quantizer is rejected when a
policy would move fewer tokens than its `group_size` at once, because a one-token
per-channel group reconstructs exactly and would silently skip quantization. Tokens that are already
old when they arrive (for example, the context imported on the first call) go straight
to the middle quantizer. A token that sat in a *quantized* local window is re-quantized
from its stored local copy when it moves; the original value is never consulted. A
local quantizer must be per-token or per-block, because local tokens arrive one at a
time. Separately for K and V, the counters satisfy:

```text
seen_tokens = sink_tokens + compressed_tokens + recent_tokens
```

Local tokens can temporarily exceed the requested minimum by less than one
block. Stored compressed chunks persist without re-encoding. Reconstructed K/V
are temporary, in original coordinates, shape, dtype and device. FAKE stores
reconstructed chunks; PACKED stores codes and decodes on read. Both use the same
encoder/decoder, rather than two different rounding implementations.

## Transforms

Each transform rotates every key/value vector along `head_dim` before quantization
and rotates it back after decoding. All are orthogonal, computed in FP32, and return
the input's original dtype.

| Transform (config) | Rotation | Used by |
|---|---|---|
| `IdentityTransform` | none (default) | most methods |
| `HadamardTransform` | normalized Hadamard `H`; `head_dim` must be a power of 2 | |
| `RandomHadamardTransform` (`seed`) | `diag(s) @ H` with random signs `s` | QuaRot |
| `RandomRotation` (`seed`) | random orthogonal matrix (QR of a Gaussian) | TurboQuant |

The random transforms derive one rotation per (layer, keys/values) from `seed`, so
runs are reproducible and keys and values never share a rotation. Rotations spread
outlier channels over all channels, which lowers per-token quantization error.

## Intentionally pending

Learned rotations, FP8/NVFP4/MXFP codecs,
nonuniform codebooks, calibration, channel boosts, outlier policies and all named
paper recipes raise `NotImplementedError` when selected. Their classes, configs
and registry entries are ready for focused implementations. KIVI's and Kitty's
buffering is implemented (`SinkAndLocalCachePolicy`, `KittyCachePolicy`); Kitty additionally
needs `ChannelBoost` (mixed-bit pages and boost indices). KVQuant requires calibrated codebooks and per-element
outliers; keys are quantized post-RoPE (no RoPE inversion). TurboQuantKV is
reserved for the MSE variant, not QJL inner-product estimation.

Mixed precision is an explicit `PrecisionPlan` extension. The working codec
rejects per-entry assignments until their packing/layout is implemented.
`QuantizedTensor` reserves format, codebook, global-scale, norm and outlier fields;
future encodings must account for all owned metadata in storage measurements.

## Scope and lifecycle

Only append-only full-history, inference K/V are supported. Sliding-window
eviction, beam reordering, rollback, batch shape changes and rewritten past tokens
need explicit policies. Length alone cannot detect a rewritten prefix.

Each independent question has one metadata dictionary and one store. The
coordinator and fixed numerical parameters can be reused, while histories live
only in metadata. The HF adapter clears the extra store in `finally`, including
failed generations. Direct callback callers must call `clear_request_state` too.

The adapter's existing context prefill remains native dense attention; its first
custom callback imports the full existing history. Direct HF callers can mark
metadata `kv_quantization_phase='prefill'` or `'decode'`; native prefill is the
default. `quantize_prefill=True` enables the generic baseline on prefill, not a
claim to reproduce any paper's prefill policy.

HF's original full-precision cache remains allocated. Store bytes describe only
additional owned tensor storage (not Python objects or temporary decoded K/V).
This implementation demonstrates numerical effects and packed storage, not total
inference memory savings or native low-bit attention speedups. Optional local
K/V error differs from the existing sparse-vs-dense output error and from final
benchmark quality. Diagnostics should be disabled for timing comparisons.

## Verification

```bash
pytest -q tests/unit/kv_quantization tests/integration/test_kv_quantization_flow.py
```

Extension order: implement one component, test its round trip and format, define
its cache/calibration policy, then enable its named method after comparing with
the reference method. A new class name alone does not establish paper fidelity.
