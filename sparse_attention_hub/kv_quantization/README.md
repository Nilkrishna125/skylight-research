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
  adapters/huggingface.py: build_custom_attention_function
    sparse_attention/base.py: SparseAttention.custom_attention (shared wrapper)
      kv_quantization/base.py: KVQuantization.process
        cache/store.py: append only unseen tokens, independently for each layer
        policies/cache/: decide which K/V chunks have aged out of the FP tail
        transforms/: forward transform
        policies/precision/: precision plan
        quantizers/: encode + pack, with per-group parameters
        cache/store.py: read encoded history + original FP sink/recent regions
        quantizers/ + transforms/: decode and inverse transform
      research_attention/base.py: ResearchAttention._attention_impl
    kv_quantization/base.py: after_attention policy extension point
  adapter's per-question finally: clear_request_state
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
│       ├── composite.py
│       ├── hadamard.py
│       ├── random_rotation.py
│       ├── rope_inverse.py
│       └── learned_rotation.py
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
│           ├── residual_policy.py
│           ├── kivi_policy.py
│           └── kitty_policy.py
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
| `transforms/` | `KVTransform`, working identity/composition; other transforms scaffolded |
| `quantizers/` | `KVQuantizer`, `QuantGranularity`, working grouped `IntQuantizer`; FP/codebook scaffolds |
| `policies/precision/` | `PrecisionPolicy`, `PrecisionPlan`, working fixed bits; mixed-bit scaffolds |
| `policies/cache/` | `CachePolicy`, working residual policy; method-specific buffering scaffolds |
| `cache/` | `QuantizedKVStore`, per-layer state, encoded tensor records and chunks |
| `metrics/quant_error.py` | Optional reconstruction errors on quantized regions; stored tensor bytes |
| `implementations/` | KIVI, QuaRotKV, TurboQuantKV, KVQuantNUQ, Kitty and NVFP4KV method declarations |

For example, `implementations/kivi.py` assembles the complete method, while
`policies/cache/implementations/kivi_policy.py` handles its buffering/compression
schedule. These have different responsibilities; quantization math is not
duplicated. Methods can share policies instead of defining a new one each time.

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

The parent public method now owns preprocessing. Attention subclasses implement
`_attention_impl` instead of overriding `custom_attention`. The optional parent
config field is keyword-only to preserve required subclass fields, requiring
Python 3.10 or newer. Existing config calls without quantization keep the OFF path.

## What is implemented

Grouped uniform INT2/4/8 quantization uses FP32 statistics and floating minimum
offsets. Asymmetric encoding is `round((x - min) / scale)`, with
`scale = (max - min) / (2**bits - 1)`. Constant groups reconstruct exactly.
The optional symmetric mode uses signed `[-qmax, qmax]`, stored as unsigned
codes. Codes are packed into uint8: four INT2 or two INT4 entries per byte.
This is a reference PyTorch implementation, not an optimized GPU kernel.

`PER_TOKEN` groups channels; `PER_CHANNEL` groups tokens. `PER_BLOCK` groups
consecutive flattened token/channel entries independently within each batch/head.
Incomplete groups repeat the last entry for statistics and discard padding on
decode. For paper reproduction, cache-policy boundaries must match that paper's
grouping instead of relying on this baseline padding convention.

The residual policy independently encodes complete K and V chunks when enough
tokens remain for their configured FP tails. It retains sink tokens in their
original dtype. Its counters satisfy, separately for K and V:

```text
seen_tokens = sink_tokens + compressed_tokens + recent_tokens
```

FP recent tokens can temporarily exceed the requested minimum by less than one
chunk. Stored compressed chunks persist without re-encoding. Reconstructed K/V
are temporary, in original coordinates, shape, dtype and device. FAKE stores
reconstructed chunks; PACKED stores codes and decodes on read. Both use the same
encoder/decoder, rather than two different rounding implementations.

## Intentionally pending

Hadamard/random/learned rotations, inverse RoPE, FP8/NVFP4/MXFP codecs,
nonuniform codebooks, calibration, channel boosts, outlier policies and all named
paper recipes raise `NotImplementedError` when selected. Their classes, configs
and registry entries are ready for focused implementations. A fixed residual
tail alone is **not** called KIVI: faithful KIVI needs its own key-flush/value-buffer
policy and prefill/compression timing. Kitty additionally needs mixed-bit page
layout and boost indices. KVQuant requires calibrated codebooks and per-element
outliers; RoPE inversion needs model-specific position context. TurboQuantKV is
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
