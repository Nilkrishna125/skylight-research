# KV quantization method implementations

This directory contains named quantization recipes and their configurations.
Shared quantizers, transforms, precision policies and cache policies live in the
parent package. The generic baseline is configured through
[`KVQuantizationConfig`](../base.py).

| Method | Configuration / class | Source | Status |
|---|---|---|---|
| [KIVI](#kivi) | `KIVIConfig` / `KIVI` | [kivi.py](kivi.py) | Implemented portable numerical/cache path |
| Kitty | `KittyConfig` / `Kitty` | [kitty.py](kitty.py) | Scaffold |
| QuaRot KV | `QuaRotKVConfig` / `QuaRotKV` | [quarot_kv.py](quarot_kv.py) | Scaffold |
| TurboQuant KV | `TurboQuantKVConfig` / `TurboQuantKV` | [turboquant_kv.py](turboquant_kv.py) | Scaffold |
| KVQuant NUQ | `KVQuantNUQConfig` / `KVQuantNUQ` | [kvquant_nuq.py](kvquant_nuq.py) | Scaffold |
| NVFP4 KV | `NVFP4KVConfig` / `NVFP4KV` | [nvfp4_kv.py](nvfp4_kv.py) | Scaffold |

Scaffold recipes inherit `PendingRecipe` and raise `NotImplementedError` when
selected. Their available building blocks do not establish a complete paper
implementation. Add each method's reference, behavior, tests and limits here as
its recipe is implemented.

## KIVI

The KIVI recipe uses a dedicated numerical quantizer and the shared request-local
cache store. The generic integer quantizer remains available for experiments.

### Reference

The implementation follows [jy-yuan/KIVI](https://github.com/jy-yuan/KIVI) at commit
`876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`:

- [quant/new_pack.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/quant/new_pack.py): grouped encoding and standalone unpack arithmetic.
- [models/llama_kivi.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/models/llama_kivi.py): residual-buffer behavior during prefill and single-token decoding.

The numerical policy reproduces the chosen released helpers. The paper does not
mandate the dtype of every intermediate operation.

### Implementation

| File | Responsibility |
|---|---|
| [Recipe](kivi.py) | `KIVIConfig`, default components and configuration validation |
| [Quantizer](../quantizers/implementations/kivi.py) | `KIVIQuantizerConfig`, native-dtype encoding, unpacking and finite-value checks |
| [Coordinator](../base.py) | Prepare attention K/V before updating and compressing the store |
| [Store](../cache/store.py) | Independent per-layer K/V histories and compressed/full-precision chunks |
| [Cache policy](../policies/cache/implementations/sink_and_local.py) | Buffered keys and rolling full-precision values |
| [Reference fixtures](../../../tests/reference/kivi) | Hash-verified authors' source, provenance, MIT license and CPU helper loader |

Defaults are INT2, group size 32 and residual length 128. Key groups span tokens
within each channel. Value groups span channels within each token.

The following excerpt is copied from the current quantizer:

```python
groups = groups.to(x.dtype)
minimum, maximum = group_min_max(groups, valid)
scale = (maximum - minimum) / (2**bits - 1)
```

Scales and minima retain the input dtype. Unlike the generic integer quantizer,
KIVI does not use directed parameter rounding through `store_param()`.
Dequantization follows the authors' standalone helper: unpacked codes become
FP16, then multiplication/addition follow the stored parameters' dtype promotion.

Keys stay in full precision until a complete residual buffer can be compressed;
values retain the newest residual window. Older compressed history is retained.
Attention receives restored previous history plus exact current-call K/V, prepared
before the persistent store is updated.

### Tests

Run from the repository root in the project's Python environment:

```bash
python -m pytest -q tests/unit/kv_quantization/test_kivi_reference.py
python -m pytest -q tests/integration/test_kivi_reference_flow.py
python -m pytest -q -m unit tests/unit/kv_quantization
python -m pytest -q -m integration tests/integration/test_kv_quantization*.py tests/integration/test_kivi_reference_flow.py
```

- The KIVI unit suite compares logical codes, scales, minima, reconstruction,
  storage ownership and streaming buffer transitions against pinned CPU helpers.
- The HF reference-flow suite uses tiny random Llama, Qwen3 and Mistral models to
  check callbacks, cache state, attention outputs and FAKE/PACKED agreement.
- Unit tests include independent packing/numerical regressions and recipe-scope
  checks. Integration tests cover cache/attention regressions, native and
  multi-architecture comparisons, component combinations and an offline benchmark
  worker. The deferred after-attention extension test is an expected failure.

These tests require no pretrained model weights or evaluation datasets.

### Scope and limits

- HF's unquantized cache remains allocated alongside our quantized store.
- Packed codes use Skylight's uint8 layout. Tests unpack both representations to
  compare logical codes with the authors' int32 layout.
- Complete groups are required. The key residual length must be a positive
  multiple of its group size; the value group size must divide the head dimension.
- Constant groups use zero scale/code. Nonconstant scale underflow and nonfinite
  input/parameters raise errors; finite output overflow follows the framework's
  saturation rule. These are explicit extensions around reference edge cases.
- BF16/FP32, INT8, kept-token policies and other component overrides are portable
  experiments. The authors' fused backend is a separate FP16 INT2/4 path.
- The shared flow supports batched current tokens, but reference streaming checks
  use native prefill followed by one-token calls. A batched call is not claimed
  numerically equivalent to sequential calls crossing buffer boundaries.
- CPU reference/callback checks do not establish fused CUDA equivalence,
  pretrained task quality, total GPU-memory savings or inference speed.
