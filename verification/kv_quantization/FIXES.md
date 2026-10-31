# Fix log

This file records the solutions applied after the [original verification](README.md).
Each new issue should get its own dated entry with the explanation, code locations,
important snippets, and test results. The original audit describes commit `449d87b`;
the fixes below are currently changes in the working tree.

Current implemented fixes: **F01, F02, F04 (during F01/F02), F05–F08 and F10–F14**.
**F03 and F09 are deferred by agreement**, for the reasons recorded below. Test counts in
each dated entry describe that run; the latest entry records the latest results.

## 31 October 2026 — F01 and F02

**F01 is fixed:** the K/V arriving for the current attention call stay in their
original precision in that call, even if storing them fills and compresses a buffer.

**F02 is fixed for the tested standard self-attention/sliding-cache flow:** each
call imports its new tokens using the query count. Our store keeps its own history
even when HF drops older entries. Context capture and mask alignment are included.

### F01: what was wrong

The old order was:

```python
store.append(...)          # Add the current tokens.
self._compress(...)        # Possibly compress them immediately.
keys, values = store.get(...)  # Reconstruct the already-updated store.
return keys, values
```

If a key buffer filled during this call, some or all current keys were returned
after quantization. In a batch, a later token could change an earlier key's shared
quantization range and therefore change an earlier attention output.

### F01: what changed

We now make the tensors for current attention **before changing the store**:

```python
previous_keys, previous_values = store.get(layer_idx, self._decode_chunk)
new_keys = keys[..., past_in_input:, :]
new_values = values[..., past_in_input:, :]

attention_keys = torch.cat([previous_keys, new_keys], dim=-2)
attention_values = torch.cat([previous_values, new_values], dim=-2)

# Update the persistent cache only after preparing those separate tensors.
store.append(layer_idx, new_keys, new_values, encode)
self._compress(...)
return attention_keys, attention_values
```

The real method also handles an empty previous history and imports completed
context on a first full-history callback. The concatenated tensors own their data:
compressing the store cannot change the tensors already prepared for attention.

Compression still runs before the attention calculation physically executes.
This is safe because attention uses its already-prepared copy. We do not need to
move every codec operation into an after-attention callback to enforce this rule.

For example, when KIVI has six buffered keys and receives two more with R=8:

```text
Current attention receives: six previous FP keys + two original new keys
The store becomes:         eight compressed keys, for subsequent calls
```

The rule also holds when sink/local regions themselves have quantizers, and when
the policy compresses every incoming token. It is enforced separately for keys
and values. Older cached regions retain the precision selected by their policy.

**Main location:** [KVQuantization.process](../../sparse_attention_hub/kv_quantization/base.py).

### F02: what was wrong

Previously, the store received a full HF tensor and calculated:

```python
new_count = incoming_length - state.seen_tokens
```

That fails when HF changes `[A,B,C,D]` into `[B,C,D,E]`: both tensors have length
four, so E was missed and our store kept returning the old values.

### F02: what changed

The self-attention caller now supplies the number of current tokens:

```python
num_new_tokens = int(queries.shape[-2])
```

This counts token positions, not query heads. The last `num_new_tokens` entries of
the incoming K/V are the current call's K/V. `process()` passes only those entries
to the store; `QuantizedKVStore.append()` no longer tries to find them by comparing
HF's length against our length.

```python
# append() now receives only new K/V.
new_count = int(keys.shape[-2])
...
state.seen_tokens += new_count
```

With HF's four-token window:

```text
HF supplies [B,C,D,E] and query count is 1.
We import E once.
Our store can retain [A,B,C,D,E], following our own storage rules.
```

Each layer keeps its own total captured-token count and chunk start positions.
These continue increasing when HF's retained tensor stops growing. Our current
policies retain all captured history while choosing its precision; HF eviction
does not trim that stored history.

**Locations:**

- [ResearchAttention.custom_attention](../../sparse_attention_hub/sparse_attention/research_attention/base.py): obtains query length.
- [SparseAttention.quantize_kv](../../sparse_attention_hub/sparse_attention/base.py): forwards count and phase.
- [KVQuantization.process](../../sparse_attention_hub/kv_quantization/base.py): separates previous history and current suffix.
- [QuantizedKVStore.append / validate_input](../../sparse_attention_hub/kv_quantization/cache/store.py): appends new tensors and checks the layer's schema.
- [LayerKVState](../../sparse_attention_hub/kv_quantization/cache/state.py): documents the independent captured-token count.

### Capturing the initial context

Query count cannot recover context that HF discarded before our first callback.
The adapter therefore enables a capture callback during context prefill:

```python
sparse_meta_data["kv_quantization_phase"] = "prefill"
sparse_meta_data["kv_quantization_capture_only"] = original_implementations

with self.enable_sparse_mode(native_prefill=True):
    context_outputs = self.model(..., sparse_meta_data=sparse_meta_data)
```

The callback copies the complete K/V into our store and delegates the calculation
to the model's original native attention function, using original K/V. Sparse
maskers are not applied during this capture. Once prefill finishes, the adapter
removes the capture-only flag and changes the phase to `decode`.

The context manager also temporarily selects the original mask factory. This
preserves the mask format expected by the native handler, then restores the
research mask factory for decode, including when prefill raises an exception.

Tests compare capture-prefill logits with native prefill for both eager attention
and SDPA, using floating-point tolerances. They also verify that an 80-token context
is retained in our store while native Mistral keeps only three past tokens for its
four-token sliding window.

Direct sliding-model callers must likewise capture prefill before starting with
a truncated history. When a late first callback exposes a sliding window, we raise
a clear error rather than claiming to recover missing context.

**Locations:** [ModelAdapterHF.process_request / get_custom_attention_function](../../sparse_attention_hub/adapters/huggingface.py).

### Matching the mask to our returned history

Our stored history and the model's attention visibility are separate decisions.
Retaining A does not make A visible if the model's sliding-window rule excludes it.

When HF supplies a mask for four keys but our returned tensor has five keys, the
new helper expands the mask onto the correct positions:

```text
Our returned keys: [A, B, C, D, E]
HF-visible keys:      [B, C, D, E]
Expanded mask:     [blocked, HF's original mask ...]
```

The helper preserves padding restrictions and additive biases, supports boolean
and floating masks, and applies causal/sliding restrictions for the current query
positions. It also handles `attention_mask=None` instead of accidentally exposing
the extra retained prefix.

**Locations:** [align_attention_mask](../../sparse_attention_hub/kv_quantization/attention_mask.py), called by
[ResearchAttention.custom_attention](../../sparse_attention_hub/sparse_attention/research_attention/base.py) before either its sparse or gated-dense calculation.

### Diagnostic errors with a sliding source

HF may now supply original values for only the final four positions while our
store holds hundreds. The optional reconstruction metric compares only encoded
positions for which that call actually supplies original K/V. It does not compare
unrelated positions or claim to have an original reference for evicted entries.
It measures the updated stored representation. The fresh suffix used for current
attention can still be exact even when its compressed stored copy has an error.

**Locations:** the metric calls in [KVQuantization.process](../../sparse_attention_hub/kv_quantization/base.py), and
[QuantErrorMetric.log](../../sparse_attention_hub/kv_quantization/metrics/quant_error.py), which accepts `start_token`.

### Calling the coordinator directly

`num_new_tokens` is now a required keyword. Standard attention callers use:

```python
attention_keys, attention_values = quantization.process(
    layer_idx, hf_keys, hf_values, metadata,
    num_new_tokens=int(queries.shape[-2]),
)
```

For codec/storage experiments with no active queries, use:

```python
stored_keys, stored_values = quantization.process(
    layer_idx, completed_history_keys, completed_history_values, metadata,
    num_new_tokens=0,
)
```

Zero means the supplied full history is already completed and may be compressed
before inspection. Existing component tests use this explicit form to continue
checking the stored representation. It is not the argument to use for a model
call that has new query tokens. Empty previous history is supported; invalid
counts, schema changes and historical rollback fail explicitly.

`quantize_prefill=False` now preserves original prefill attention inputs while
still capturing the cache. With `True`, a direct chunked-prefill caller may use
reconstructed earlier chunks; current-call K/V still stay exact. The adapter's
capture-only prefill delegates to native attention in either case.

### Tests added and updated

| File | What it checks |
|---|---|
| [test_current_call_history.py](../../tests/unit/kv_quantization/test_current_call_history.py) | Current K/V exactness across buffer flushes, separately compressed storage, independent tensor ownership, quantized sink/local regions, full-quantization policy, rolling HF history, layer isolation, count validation and metric alignment |
| [test_attention_mask.py](../../tests/unit/kv_quantization/test_attention_mask.py) | Causal/sliding visibility, boolean/additive/absent masks, multiple batches, padding and preserved biases |
| [test_kv_quantization_sliding.py](../../tests/integration/test_kv_quantization_sliding.py) | Real Mistral with long prefill, eager/SDPA, independent questions, new-token ingestion, cleanup, five-codec FAKE/PACKED logit parity, two-sequence left padding and causal batched questions at a KIVI flush |
| [test_audit_regressions.py](test_audit_regressions.py) | F01/F02 now have passing checks instead of expected failures; F02 checks independent retention rather than requiring our history to equal HF's window |

The original numerical component tests retain their assertions. Their direct
coordinator calls now explicitly pass zero active tokens when inspecting stored
history. The ordinary adapter and attention regression suites are also run.

### Related findings and scope

- **F03:** its current-step key-flush timing reproduction now passes as a consequence
  of preparing attention before updating storage. Complete paper/kernel numerical
  parity remains unverified; F03 is not fully closed.
- **F04:** query-phase forwarding and prefill capture were implemented as dependencies
  of these fixes. Its phase-forwarding reproduction now passes; the new prefill
  contract is documented above.
- **F13:** typing invariants in touched files were made explicit and an unused test
  import removed. This does not close formatting/type issues elsewhere in the audit.
- **F05–F09 and F11:** their seven remaining expected-failure cases stay in the audit.
  F10/F12/F14 also retain work outside this change. These fixes do not establish
  total GPU-memory savings or native low-bit speedups.

Supported here: ordinary causal self-attention, contiguous new K/V suffixes, and
append-only or native rolling HF caches whose prefill is captured. HF mask visibility
is preserved while our storage remains independent. Static padded caches, beam
reordering, speculative rollback, and a future policy that evicts/reorders our own
stored positions need separate contracts/tests. GPU kernels and trained-model
quality were not measured.

### Verification record

The executable results are saved under `results/f01_f02_*`. They use the audit
environment recorded in [environment.json](results/environment.json); models and
tokenizers are generated locally, without model/dataset downloads.

```bash
AUDIT_PY=/tmp/skylight-audit-venv/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/skylight-audit-mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"$AUDIT_PY" -m pytest -p pytest_mock -q \
  tests/unit/kv_quantization \
  tests/integration/test_kv_quantization_flow.py \
  tests/integration/test_kv_quantization_sliding.py \
  tests/unit/adapters \
  tests/unit/sparse_attention/test_base.py \
  tests/unit/sparse_attention/test_configs_and_factories.py \
  tests/unit/sparse_attention/research_attention/test_base.py \
  verification/kv_quantization
```

| Check | Result | Saved output |
|---|---|---|
| Quantization, adapter/attention regressions and audit checks above | **565 passed, 7 expected failures** | [Targeted tests](results/f01_f02_tests.log) |
| Broader unit run, excluding the expensive DoubleSparse correctness file | **1,152 passed, 8 skipped, 1 failed** | [Broader unit tests](results/f01_f02_broader_unit.log) |
| Mypy on quantization and the touched attention/adapter modules | **53 source files passed** | [Type checks](results/f01_f02_mypy.log) |
| Black, isort and flake8 on 21 edited Python files; local documentation links | **Passed** | [Style/link checks](results/f01_f02_style.log) |

The broader failure is
`TestAdaptiveSamplingMasker.test_get_std_estimate_using_base_sample`, in unchanged
adaptive-sampling code. The original audit also observed this failure. It remains
open; the broader suite is not fully green.

The seven expected failures reproduce other open findings; they are not successful
correctness assertions. These checks ran on CPU with locally generated models.
CUDA behavior, GPU-memory savings, speedups and trained-model quality remain
unverified. `git diff --check` also passed.

## 31 October 2026 — Decisions on F03 and F04

**F03 — deferred by agreement.** This finding concerns reproducing the KIVI paper
or its released code, whose compression timing differs. We will decide the target
and verify full numerical parity when working on that paper implementation. It
is not being treated as another generic flow defect to fix now. The flush-timing
regression already passes after F01; that alone does not establish complete KIVI
reference parity.

**F04 — fixed as part of F01/F02.** `SparseAttention.quantize_kv()` now forwards
`kv_quantization_phase` from metadata into `KVQuantization.process()`. The adapter
sets `prefill` during initial context capture and `decode` afterward. Its phase
regression passes. Original context-prefill attention remains native while our
store captures K/V; `quantize_prefill=True` does not enable quantized context
prefill through the adapter. See the lifecycle explanation above.

We will revisit masks based on our own cache's visibility policy after the
quantization-only benchmarking stage. The current HF-visibility alignment stays
as described in the F02 entry.

## 31 October 2026 — F05: exclude preserved tokens within the codec

### What was wrong

The optional generic `OutlierTokenPolicy` preserves selected tokens exactly.
Previously, the coordinator saved their originals and replaced them with nearby
non-kept tokens before quantization. That neighbour could belong to another
numerical group:

```text
Group 1: [100, 100, 100, 100]
Group 2: [9999, 1, 2, 3]       # Preserve 9999 exactly.

Old temporary Group 2: [100, 1, 2, 3]
Range used: 1–100, although the values needing quantization span only 1–3.
```

The preserved token was restored correctly, but its temporary replacement made
the remaining values less precise. Choosing another neighbour globally cannot
reliably respect different codecs' numerical groups.

### The rule we agreed to implement

This is an **optional generic component contract**, not a rule imposed on every
paper implementation:

- A precision policy chooses the tokens to preserve and returns `keep_tokens`.
- The coordinator saves those originals and restores them after dequantization
  and the inverse transform, at the original positions.
- A supporting codec excludes those positions from its encoding statistics,
  using its own grouping and scaling rules.
- A paper recipe selects this policy only if its intended method calls for it.
  Different paper-specific outlier treatment can use different components.

No token is removed or moved to another group. Kept positions still have disposable
encoded placeholders; reconstruction overwrites them with the saved originals.
This change does not claim a reduction in payload size for those placeholders.
The separate F06 FAKE-byte-count issue was still open in this run; its subsequent
fix is recorded below.

### What changed, with locations and snippets

**1. Keep the exclusion mask when calling the codec.**

[KVQuantization._encode_chunk](../../sparse_attention_hub/kv_quantization/base.py)
still saves `kept_index` and `kept_values`. It now passes the original precision
plan to the codec, including `keep_tokens`:

```python
# Save the original kept values as before.
encoded = quantizer.quantize(transformed, plan, context=context)
```

The old `fill_kept_tokens()` helper and the clearing of `plan.keep_tokens` were
removed. `restore_kept_tokens()` continues restoring originals after the inverse
transform, so a saved token is exact in the model's original coordinates.

**2. Group data and validity through the same layout.**

[group_for_quantization / group_min_max](../../sparse_attention_hub/kv_quantization/quantizers/common.py)
provide shared helpers. The codec supplies its own granularity and group size:

```python
groups, layout, valid = group_for_quantization(
    x, self.granularity, self.config.group_size, plan
)
lo, hi = group_min_max(groups, valid)
```

The validity mask excludes kept entries and padding. Kept/padded data slots are
zeroed for safe encoding. INT min/max explicitly ignores these zeros; otherwise
zero itself could incorrectly widen a positive-only or negative-only range.
For a group with no valid entries, INT uses a dummy `[0, 0]` range and a safe
nonzero scale. All its original entries are restored later.

With no `keep_tokens` mask, grouping and statistics follow the previous path.
Shape, boolean dtype and device checks reject malformed masks.

**3. Each working codec handles its own statistics.**

| Codec/location | Exclusion behavior |
|---|---|
| [INT](../../sparse_attention_hub/kv_quantization/quantizers/implementations/integer.py) | Masked group min/max, including symmetric and mixed-channel-bit encoding |
| [FP8](../../sparse_attention_hub/kv_quantization/quantizers/implementations/fp8.py) | Kept slots contribute zero to group absolute maxima |
| [FakeNVFP4](../../sparse_attention_hub/kv_quantization/quantizers/implementations/fake_nvfp4.py) | Exclusion applies to both block maxima and the chunk-wide maximum |
| [MXFP](../../sparse_attention_hub/kv_quantization/quantizers/implementations/mxfp.py) | Kept slots do not affect block maxima or scale exponents |
| [NonUniform](../../sparse_attention_hub/kv_quantization/quantizers/implementations/non_uniform.py) | Kept vectors become zero placeholders before Gaussian norm calculation or fixed-table lookup; encoding does not recalibrate tables |

Codecs opt in explicitly using `check_input(..., allow_keep_tokens=True)`.
The shared validator rejects kept-token plans by default for codecs that have
not implemented the contract. [KVQuantizer.quantize](../../sparse_attention_hub/kv_quantization/quantizers/base.py)
documents that raw codec dequantization returns placeholders; the coordinator
owns exact-token restoration. The plan and policy contracts are documented in
[precision/base.py](../../sparse_attention_hub/kv_quantization/policies/precision/base.py)
and [outlier_token.py](../../sparse_attention_hub/kv_quantization/policies/precision/implementations/outlier_token.py).

### Tests and acceptance

The normal suite now includes [test_outlier_exclusion.py](../../tests/unit/kv_quantization/test_outlier_exclusion.py):

- Hold the keep mask fixed and change kept values from `10000` to `-1000000`.
  All encoded codes and parameters must remain identical across the working
  codec variants, including NVFP4's global scale and Gaussian norms.
- Check independent expected INT parameters/output for the original two-group
  example, plus positive-only and negative-only surviving ranges.
- Cover multiple batches/heads, group boundaries, partial final groups, all-kept
  groups/heads, and completely kept histories.
- Verify exact original restoration and FAKE/PACKED agreement, including a
  rotation before encoding and mixed 2/4/8-bit channel plans.
- Verify an empty exclusion mask preserves ordinary numerical results, and
  malformed/unsupported masks fail explicitly.

The old neighbour-selection test was removed because that helper is no longer
the intended behavior. The original component assertions remain; its old
"quantizers reject kept-token plans" test now checks the supported empty-mask
behavior. The [F05 audit regression](test_audit_regressions.py) now verifies actual
codec parameters and reconstructed ordinary values, replacing the old helper-only
reproduction and removing its expected-failure marker.

### Verification record

| Check | Result | Saved output |
|---|---|---|
| New F05 cases, precision-policy suite and audit regressions | **150 passed, 6 expected failures** | [Focused tests](results/f05_focused_tests.log) |
| Full targeted quantization, adapter/attention and audit suite | **621 passed, 6 expected failures** | [Regression tests](results/f05_tests.log) |
| Mypy on the quantization package | **50 source files passed** | [Type checks](results/f05_mypy.log) |
| Black, isort and flake8 on the 13 F05 Python files | **Passed** | [Style checks](results/f05_style.log) |

There are 56 new F05 unit cases. The six remaining expected failures concern
F06, F07, F08 (two codecs), F09 and F11. These are open findings, not passing
correctness assertions. F03 reference-parity work and the later masking redesign
remain deferred as stated above. The earlier F01/F02 result table remains a
historical record of that run; this change did not repeat the broader unit sweep.

Using the same audit environment and environment exports shown in the F01/F02
entry, reproduce the focused checks with:

```bash
"$AUDIT_PY" -m pytest -p pytest_mock -q \
  tests/unit/kv_quantization/test_outlier_exclusion.py \
  tests/unit/kv_quantization/test_precision_policies.py \
  verification/kv_quantization/test_audit_regressions.py
```

The full regression command is the same directory/file selection shown in the
F01/F02 entry; it automatically includes the new test file. Type checks used:

```bash
"$AUDIT_PY" -m mypy sparse_attention_hub/kv_quantization \
  --follow-imports=silent --ignore-missing-imports
```

Local documentation links and `git diff --check` passed. All checks used CPU;
CUDA behavior, native low-bit kernels and paper-level numerical parity were not
established by this change.

## 31 October 2026 — F06: release redundant originals in FAKE storage

### What was wrong

FAKE mode stores a floating-point reconstruction of the quantized values. Exact
kept tokens are already restored inside that reconstruction, but the chunk also
retained their separate `kept_values` and `kept_index` tensors. The byte counter
returned the reconstruction's size without counting those extra allocations.

The audit's one-outlier example owned 2,048 bytes of reconstructed K/V plus
128 bytes of copied originals and a 4-byte index. It reported only 2,048 bytes,
although those tensors actually occupied 2,180 bytes.

### What changed

[KVQuantization._encode_chunk](../../sparse_attention_hub/kv_quantization/base.py)
now releases the two extra tensors **after** reconstructing, applying the inverse
transform and restoring the exact originals:

```python
chunk.dequantized = restore_kept_tokens(
    transform.inverse(quantizer.dequantize(encoded), context, saved),
    chunk,
).detach().clone()
chunk.kept_index = None
chunk.kept_values = None
```

These assignments run only in FAKE mode. PACKED chunks still retain their codes,
kept indices and original values, because future decoding needs them to restore
the exact tokens.

| Original audit example | Reported bytes | Actual stored tensor bytes |
|---|---:|---:|
| Before | 2,048 | 2,180 |
| After | 2,048 | 2,048 |

The counter in [QuantizedKVStore._chunk_bytes](../../sparse_attention_hub/kv_quantization/cache/store.py)
does not need a formula change: the redundant allocations are gone, and a FAKE
chunk's reconstructed tensor now contains its complete payload. The field-lifetime
rule is documented in [StoredChunk](../../sparse_attention_hub/kv_quantization/cache/block.py).
The preserved K/V values and their precision do not change.

This count covers stored tensor payloads and per-record quantizer parameters.
It does not include Python objects, shared components, context/transform state,
the native HF cache, temporary reconstruction or total device allocation. Releasing
a reference also does not imply that a GPU allocator immediately returns memory
to the operating system.

### Tests and acceptance

The normal suite now includes [test_store_accounting.py](../../tests/unit/kv_quantization/test_store_accounting.py):

- Reproduce the original example and assert both actual storage and the reported
  size are exactly 2,048 bytes, with the preserved token still exact.
- Independently count distinct backing tensor allocations and compare with
  `store.stats()` in FAKE and PACKED modes, with and without outliers.
- Cover FP16, BF16 and FP32, multiple batches/heads, and a rotation before encoding.
- Confirm PACKED retains the original-value/index tensors it needs, while FAKE
  releases them after restoration.
- Modify the caller's input buffers and read the store again: the stored originals
  must remain exact and independent of those buffers.
- Append another attention batch and verify previously kept tokens still restore
  correctly, current-call tokens stay exact and byte counts remain correct.

The F05 all-kept-codec tests still verify exact FAKE/PACKED reconstruction; their
metadata assertions now expect extra original-value tensors only in PACKED mode.
The [F06 audit regression](test_audit_regressions.py) keeps its original independent
byte-count assertion and no longer has an expected-failure marker.

### Verification record

| Check | Result | Saved output |
|---|---|---|
| Store-accounting tests, F05 cases, precision policies and audit regressions | **165 passed, 5 expected failures** | [Focused tests](results/f06_focused_tests.log) |
| Full targeted quantization, adapter/attention and audit suite | **636 passed, 5 expected failures** | [Regression tests](results/f06_tests.log) |
| Mypy on the quantization package | **50 source files passed** | [Type checks](results/f06_mypy.log) |
| Black, isort and flake8 on the five F06 Python files | **Passed** | [Style checks](results/f06_style.log) |

There are 14 new normal unit cases for F06. Its original audit reproduction also
now passes. The five expected failures still reproduce open findings: F07, F08
(two codecs), F09 and F11. Earlier result tables retain their historical counts.

With the audit environment and exports from the F01/F02 entry, reproduce the
focused checks with:

```bash
"$AUDIT_PY" -m pytest -p pytest_mock -q \
  tests/unit/kv_quantization/test_store_accounting.py \
  tests/unit/kv_quantization/test_outlier_exclusion.py \
  tests/unit/kv_quantization/test_precision_policies.py \
  verification/kv_quantization/test_audit_regressions.py
```

The full regression command is the same directory/file selection in the F01/F02
entry, which includes the new test file automatically. Type checks used the same
command shown in the F05 entry. Local documentation links and `git diff --check`
also passed. Checks ran on CPU; total GPU-memory behavior was not measured.

## 31 October 2026 — F07: freeze calibrated tables after use

### What was wrong

The calibrated nonuniform codec stores small integer codes that index a learned
table. For example, code 2 might mean 0.5 under table A. Recalibration or loading
another file could replace A with table B, where code 2 means 100.5. Existing
PACKED entries would then silently reconstruct different K/V even though their
stored codes had not changed. The records previously identified only the layer
and K/V kind, not the table contents.

### The agreed lifecycle

We chose a **fixed-table contract** for the existing calibrated codec:

```text
Prepare all required tables -> first successful encoding -> tables stay frozen
```

`calibrate()` and `load()` may prepare or replace tables before encoding starts.
The first successful calibrated encoding freezes the whole quantizer instance.
Later recalibration or loading raises a clear error directing the caller to use
a fresh quantizer and cache. Existing entries are not rewritten or requantized.

The rule applies to both FAKE and PACKED encoding. Request cleanup does not thaw
the codec: cached records may still exist outside the request, and a used codec
keeps the same table meaning for its lifetime. New requests can continue using
the existing frozen tables normally. Preparing another model/table configuration
requires a new quantizer. Gaussian Lloyd-Max mode does not use this lifecycle;
its numerical path is unchanged.

### Changes, locations and snippets

All production changes for this finding are in
[NonUniformQuantizer](../../sparse_attention_hub/kv_quantization/quantizers/implementations/non_uniform.py).

**1. Freeze after successful encoding, not after an unsuccessful attempt.**

`_quantize_calibrated()` validates the requested layer/kind and channel count,
encodes the values, then records the table identity and freezes the instance:

```python
encoded = self._record(x, packed, count, None, mode="calibrated", key=key)
encoded.metadata["table_fingerprint"] = self._table_fingerprints[key]
self._tables_frozen = True
return encoded
```

Missing context/tables, incompatible channel counts, invalid inputs/plans or
encoding failure leave a fresh codec unfrozen, so preparation can be corrected.
The public `tables_frozen` property exposes the state for inspection.

**2. Reject table changes after use.**

Both `calibrate()` and `load()` call `_check_tables_mutable()` before changing
anything. They also reject use in Gaussian mode, where calibrated tables are not
part of the method. `save()` remains allowed for prepared tables after freezing.

```python
codec.calibrate(samples_by_layer_and_kind)  # Prepare all needed tables first.
encoded = codec.quantize(kv, context=context)  # Freezes calibrated tables.
codec.calibrate(other_samples)  # RuntimeError: use a fresh quantizer/cache.
codec.load("another_table.pt")  # Same error; no table is changed.
codec.save("prepared_tables.pt")  # Allowed; does not unfreeze the codec.
```

The codec cannot know the whole model's required layers in advance. Prepare those
layer/K/V-kind tables before inference; attempting to encode a missing table fails
explicitly. Multiple valid preparation calls may merge tables before the freeze.

**3. Protect table ownership.**

The live mapping is private (`_tables`). The public `tables` property returns a
read-only mapping of detached tensor copies. Assigning a dictionary entry or
replacing the property fails; modifying an inspected tensor changes that copy,
not the live table. All installed tables are owned, contiguous FP32 CPU copies.
The F05 fixtures now use public `calibrate()` calls instead of dictionary writes.

**4. Check the table identity when decoding.**

Validated table contents and shape receive a SHA-256 fingerprint once during
preparation. Each calibrated record references that fingerprint in metadata.
`dequantize()` checks it, the bit width and channel count before table lookup.
Passing a record to a different quantizer with different tables therefore raises
an error, instead of silently returning different numbers.

A fresh codec loaded with identical tables can decode a matching record. That
successful decode freezes its tables too, since records now depend on them.
This is an identity check, not adaptive codebook versioning: a codec still keeps
only one prepared table per layer/kind. No full table is duplicated into every
cached record, and the stored code payload format is unchanged. Calibrated records
without the new fingerprint are rejected rather than decoded against an unverified
table. Existing valid table files still use the same `bits`/`tables` schema.

**5. Validate preparation before committing changes.**

- Table files require a matching integer bit width and a nonempty table mapping.
- Names must identify a non-negative layer index and `key` or `value`.
- Each table must be a dense floating `[channels, 2**bits]` tensor with at least
  one channel, finite FP32 levels and nondecreasing level order. Equal levels are
  allowed for constant calibration data.
- Calibration samples must be nonempty, floating and finite after FP32 conversion;
  iteration counts must be positive integers.
- The whole update is prepared and validated before installing any table. An
  invalid later entry cannot partially replace previously prepared tables.

A lock serializes table preparation with encoding/decoding so a table update
cannot race with the first freeze. Worker serialization preserves table contents,
fingerprints and freeze state while recreating that lock on deserialization.
This change establishes ownership/lifetime for the generic calibrated codec;
it does not implement a new paper recipe or coordinator-level calibration.

### Tests and acceptance

The new normal suite is
[test_calibrated_table_lifecycle.py](../../tests/unit/kv_quantization/test_calibrated_table_lifecycle.py).
It covers expected packed codes and reconstructed values, reload/recalibration
rejection, mutation through inspection copies, failure before first successful
encoding, save/load after freezing, decoding with matching/different tables,
request cleanup, invalid artifacts/samples, multi-entry update atomicity, constant
tables, worker serialization and a controlled race between first encoding and
table replacement.

The [F07 audit regression](test_audit_regressions.py) now passes without its expected-
failure marker. Its assertion was strengthened: it verifies old decoded values
remain unchanged even when the recalibration attempt is rejected. Existing
nonuniform numerical and FAKE/PACKED tests retain their assertions.

### Verification record

| Check | Result | Saved output |
|---|---|---|
| New lifecycle cases, existing nonuniform/F05 cases and audit regressions | **218 passed, 4 expected failures** | [Focused tests](results/f07_focused_tests.log) |
| Full targeted quantization, adapter/attention and audit suite | **707 passed, 4 expected failures** | [Regression tests](results/f07_tests.log) |
| Mypy on the quantization package | **50 source files passed** | [Type checks](results/f07_mypy.log) |
| Black, isort and flake8 on the four F07 Python files | **Passed** | [Style checks](results/f07_style.log) |

There are 70 new normal unit cases for this lifecycle. The four remaining expected
failures cover F08 (two codecs), F09 and F11; other audit work without expected-
failure cases is still tracked in the original report. F03 and the later masking
redesign remain deferred. Earlier result tables describe their historical runs.

Using the same audit environment and exports recorded in the F01/F02 entry,
reproduce the focused checks with:

```bash
"$AUDIT_PY" -m pytest -p pytest_mock -q \
  tests/unit/kv_quantization/test_calibrated_table_lifecycle.py \
  tests/unit/kv_quantization/test_non_uniform_quantizer.py \
  tests/unit/kv_quantization/test_outlier_exclusion.py \
  verification/kv_quantization/test_audit_regressions.py
```

The full regression command uses the same directory/file selection in the F01/F02
entry. Type checks use the command in the F05 entry. Local documentation links and
`git diff --check` passed. These checks ran on CPU, with standard Python pickling
for the worker-copy checks; GPU, distributed Ray execution and trained-model
paper parity were not measured.

## 31 October 2026 — F08: saturate finite output overflow, reject invalid numbers

### What was wrong

FP16 can represent finite values up to 65,504 in magnitude. Stored quantization
parameters can round a reconstruction slightly beyond that range while the
calculation remains finite in FP32. The final unchecked FP16 conversion then
produces infinity.

The original INT2 example uses input `[-65504, +65504]`. Its stored offset is
`-65504` and its upward-rounded scale is `43680`. Code 3 reconstructs as:

```text
-65504 + 3 × 43680 = 65536   (finite FP32)
65536 converted to FP16 = inf
```

The FP8 reproduction reconstructs approximately `±65520` in FP32 and similarly
overflows on conversion to FP16. Both now return the finite FP16 boundary values.

### The agreed rule

- **Finite reconstruction outside the output dtype's range:** clamp to that
  dtype's finite minimum/maximum, then convert.
- **NaN or infinity in input, parameters or reconstruction:** raise `ValueError`
  with a numerical error message. Invalid numbers are not replaced with zeros
  or silently clamped.
- Values within the output range retain ordinary conversion/rounding behavior.
  The clamp limits follow the actual destination dtype, not a hardcoded FP16
  constant.

This is the generic reconstruction-boundary contract. It does not change group
selection, codebook levels, stored-code packing or ordinary finite parameter
rounding. Paper-reference boundary behavior still needs to be documented when
validating a particular recipe.

### Changes, locations and snippets

**1. Shared finite checks and conversion.**

The new [numerics.py](../../sparse_attention_hub/kv_quantization/numerics.py) contains
`require_finite()` and `finite_cast()`. The conversion helper implements:

```python
require_finite(x, name="Reconstructed values")
working = x if x.dtype == torch.float64 else x.float()
limits = torch.finfo(dtype)
if limits.max < torch.finfo(working.dtype).max:
    working = working.clamp(min=limits.min, max=limits.max)
return working.to(device=device, dtype=dtype).contiguous()
```

FP16/BF16 inputs widen before saturation: BF16 cannot exactly represent FP16's
65,504 limit and would round it to 65,536 if clamped directly in BF16. FP64 inputs
remain FP64 until the clamp, so a finite wide value does not overflow in an early
FP32 conversion. Widening conversions skip unnecessary narrow-range limits.

**2. All working codec reconstruction boundaries use the helper.**

- [INT](../../sparse_attention_hub/kv_quantization/quantizers/implementations/integer.py),
  including mixed channel widths.
- [FP8](../../sparse_attention_hub/kv_quantization/quantizers/implementations/fp8.py).
- [FakeNVFP4](../../sparse_attention_hub/kv_quantization/quantizers/implementations/fake_nvfp4.py).
- [MXFP](../../sparse_attention_hub/kv_quantization/quantizers/implementations/mxfp.py).
- [NonUniform](../../sparse_attention_hub/kv_quantization/quantizers/implementations/non_uniform.py),
  for Gaussian and calibrated modes.

Their final conversion now calls `finite_cast(restored, q.dtype, device=q.device)`.
The [KVQuantizer contract](../../sparse_attention_hub/kv_quantization/quantizers/base.py)
documents this rule. Native NVFP4 remains a scaffold.

**3. Protect the conversion after inverse rotation too.**

A rotated codec can reconstruct in FP32 successfully but overflow when the inverse
transform restores the model's FP16 dtype. The inverse paths in
[Hadamard](../../sparse_attention_hub/kv_quantization/transforms/implementations/hadamard.py),
[RandomHadamard](../../sparse_attention_hub/kv_quantization/transforms/implementations/random_hadamard.py)
and [RandomRotation](../../sparse_attention_hub/kv_quantization/transforms/implementations/random_rotation.py)
now use `finite_cast()` at that boundary. [Identity](../../sparse_attention_hub/kv_quantization/transforms/implementations/identity.py)
checks finiteness while preserving its unchanged input tensor. The generic
[inverse-transform contract](../../sparse_attention_hub/kv_quantization/transforms/base.py)
states the rule.

**4. Reject invalid data before it enters pending/full-precision storage.**

[QuantizedKVStore.append](../../sparse_attention_hub/kv_quantization/cache/store.py)
checks newly captured keys and values before storing them. Invalid new values
therefore fail even when the policy leaves them uncompressed in a local buffer.
Checks run on appended tokens, not repeatedly over the store's complete history.

[store_param](../../sparse_attention_hub/kv_quantization/quantizers/common.py)
also checks parameter values for NaN/infinity before rounding. A finite scale or
offset too large for its configured parameter dtype still raises its existing
error: clipping coding parameters would change the encoding itself. Select a
wider parameter dtype when necessary.

### Tests and acceptance

The normal suite now includes
[test_finite_reconstruction.py](../../tests/unit/kv_quantization/test_finite_reconstruction.py),
covering the exact original INT2/FP8 failures, both output signs, FP16/BF16/FP32
limits, ordinary/widening casts, zeros/constants/tiny groups, symmetric and mixed-
width INT, invalid inputs even when kept, invalid decoded values/FP8 codes,
calibrated levels beyond the output range, inverse-rotation overflow, FAKE/PACKED
agreement and rejection before uncompressed cache storage.

The two [F08 audit cases](test_audit_regressions.py) no longer use expected-failure
markers. They now require successful finite saturation and exact boundary outputs;
throwing an error on those valid finite examples is no longer accepted.

### Scope and remaining runner work

F09 and F10 are unchanged: the after-attention hook is still disconnected, and
benchmark workers still omit quantization diagnostic metrics. They remain separate
findings. This change also does not introduce per-question benchmark recovery or
general attention/logit NaN detection. Numerical exceptions still propagate to the
existing caller/worker error boundary; finer sample-level continuation is separate
benchmark-runner work. An internal FP32 operation producing NaN/infinity raises;
saturation applies only while the reconstruction is still finite.

### Verification record

| Check | Result | Saved output |
|---|---|---|
| New finite-reconstruction cases, codec/transform tests and audit regressions | **276 passed, 2 expected failures** | [Focused tests](results/f08_focused_tests.log) |
| Full targeted quantization, adapter/attention and audit suite | **808 passed, 2 expected failures** | [Regression tests](results/f08_tests.log) |
| Mypy on the quantization package | **51 source files passed** | [Type checks](results/f08_mypy.log) |
| Black, isort and flake8 on the 16 F08 Python files | **Passed** | [Style checks](results/f08_style.log) |

There are 99 new normal unit cases. Both original F08 reproductions now pass.
The two remaining expected failures reproduce F09 and F11; F10 and other open
findings without such cases remain tracked in the audit. Earlier result tables
retain their historical counts.

Using the audit environment and exports from the F01/F02 entry, reproduce the
focused checks with:

```bash
"$AUDIT_PY" -m pytest -p pytest_mock -q \
  tests/unit/kv_quantization/test_finite_reconstruction.py \
  tests/unit/kv_quantization/test_fp_quantizers.py \
  tests/unit/kv_quantization/test_integer_quantizer.py \
  tests/unit/kv_quantization/test_transforms.py \
  verification/kv_quantization/test_audit_regressions.py
```

The full regression command uses the same directory/file selection in the F01/F02
entry. Type checks use the command in the F05 entry. Local documentation links and
`git diff --check` also passed. Checks ran on CPU; GPU kernels, trained-model
quality and performance were not evaluated.

## 31 October 2026 — F09 deferred: no current method needs the finish hook

The `after_attention()` extension is still disconnected. We have explicitly
deferred connecting it because all current production cache policies schedule
compression before attention. None of the runnable methods needs a finish hook
for its present implementation. Other named method classes are scaffolds; their
eventual paper-specific timing must be checked when implementing them.

F01 already prepares separate attention tensors containing reconstructed previous
history plus exact new K/V before the persistent store is compressed. Updating
the store beforehand therefore cannot change that call's prepared attention K/V.
An extra callback is not needed to protect the current-call precision contract.

This is a deferred, unsupported extension, not a completed fix and not a claim
that every future paper can use the same timing. Revisit it if a selected method
requires post-attention work or decisions based on attention outputs. Such a
method may also need a richer callback contract, since the current hook does not
receive attention weights automatically.

The [F09 audit reproduction](test_audit_regressions.py) remains a strict expected
failure, with its reason updated to record the deliberate deferral. The hook's
production behavior has not changed.

## 31 October 2026 — F10–F14: benchmark delivery, test selection and compatibility

### F10 — Record the requested quantization diagnostics

**Problem:** the worker enabled only research-attention metrics. Quantization
could compute K/V errors and store bytes without the worker saving those events.

**Changes:**

- [executor._get_enabled_micro_metrics](../../benchmark/executor.py) selects the
  three quantization metric IDs when the job's quantization config has
  `error_metric=True`. Existing research-attention defaults are preserved.
- `BenchmarkExecutor` accepts `micro_metric_logger_enabled_metrics`: `None` uses
  those defaults, `[]` disables all micro metrics, and a list selects explicit
  registered names. The setting is forwarded to each worker. Unknown metric
  names and quantization-metric requests without `error_metric=True` fail clearly.
- [KVQuantization.process](../../sparse_attention_hub/kv_quantization/base.py)
  checks which diagnostics are enabled before calculating/logging them. Disabled
  reconstruction errors do not trigger the extra stored reconstruction or error
  calculation. Selecting only store bytes avoids the error calculations too.
- Each worker writes its effective selection, record limit and sampling factor
  to `micro_metrics_config.json`. Events are saved in `micro_metrics.jsonl`.
- A job's pending diagnostics are flushed in worker cleanup, including after
  inference failure, before the singleton logger changes output directory. GPU
  cleanup still runs. If writing the log itself fails, the worker records the
  logging error and discards the unwritable remainder rather than carrying it
  into another job's output.

The worker now configures logging through:

```python
enabled_metrics = _get_enabled_micro_metrics(
    stub.sparse_attention_config,
    micro_metric_logger_enabled_metrics,
)
```

For timing runs, use `micro_metric_logger_enabled_metrics=[]`. Record sampling
and limits control saved events; they do not themselves disable all diagnostic
computation. This change does not add per-question numerical-error recovery.

**Tests:** [test_quantization_worker_metrics.py](../../tests/unit/benchmark/test_quantization_worker_metrics.py)
runs the actual worker function on CPU, replacing only GPU allocation/context and
model construction. It uses real quantization and reads actual JSONL records.
Cases cover all three diagnostics, OFF/default/subset selections, disabled
computation, saved logging configuration, failure followed by another job, record
and sampling limits, invalid selections and executor configuration ownership.
The F02 suffix-diagnostic test now explicitly enables its observed metrics while
retaining its alignment and finite-value assertions.

### F11 — Export readable, stable configuration values and method identity

**Problem:** generic object introspection ran before Enum handling. Values such
as `per_token` and `fp16` became dictionaries of Python internals, including
runtime-specific representations. Recursive `dataclasses.asdict()` also discarded
the concrete types of nested configurations.

[make_serializable](../../benchmark/utils/data.py) now handles Enum instances first:

```python
if isinstance(obj, Enum):
    return make_serializable(obj.value)
```

Dataclass fields are visited individually, retaining the concrete class at each
level using a stable `__type__` name:

```json
{
  "__type__": "sparse_attention_hub.kv_quantization.implementations.kivi.KIVIConfig",
  "key_quantizer": {
    "__type__": "sparse_attention_hub.kv_quantization.quantizers.implementations.integer.IntQuantizerConfig",
    "granularity": "per_channel",
    "scale_dtype": "fp16"
  }
}
```

This excerpt omits other fields for brevity; the export retains the fields.
Python types/functions use qualified names, paths/dtypes/devices use strings,
Enum dictionary keys are normalized and sets get deterministic ordering. Opaque
unsupported objects expose their stable class identity instead of a default repr
containing an address. Existing tensor/array string summaries remain summaries;
this exporter is not a checkpoint or lossless arbitrary-object serializer.

**Tests:** [test_config_serialization.py](../../tests/unit/benchmark/test_config_serialization.py)
checks Enum variants, nested dataclass subclass identity, repeated equivalent
configurations, collections, types/functions and ordinary special values. A real
`Benchmark.run_benchmark()` with an offline fixture verifies the saved `config.json`,
`raw_results.csv` and `metrics.json`. The original F11 audit case now passes without
its expected-failure marker.

### F12 — Make quantization category filters select the complete suites

The nine original unit modules now declare `pytestmark = pytest.mark.unit`, and
the original local-Qwen3 integration module declares `pytest.mark.integration`.
The newer files already carried their category markers. Directory-based test
selection continues working.

[test_quantization_test_categories.py](../../tests/unit/benchmark/test_quantization_test_categories.py)
performs real pytest collection in fresh subprocesses. For the quantization unit
directory and both local-model integration files, it compares unfiltered node IDs
with category-filtered node IDs and requires identical, nonempty sets. These
checks specifically cover the documented quantization commands; unrelated legacy
test-category conventions are not redefined.

### F13 — Pass the configured quality gates

Earlier fixes resolved the quantization typing errors and the unused transform
test import. The remaining Black/isort files were formatted using the pinned
Black 22.10.0 and isort 5.12.0 settings. The complete CI source/test/script scope
was checked, alongside the modified benchmark files for formatting.

The full-package type check additionally exposed two pre-existing missing
annotations in [SocketTopKMasker](../../sparse_attention_hub/sparse_attention/research_attention/maskers/fixed/implementations/socket_top_k.py).
Its configuration post-init now declares `-> None`, and `add_mask()` annotates
`**kwargs: Any`. Runtime behavior is unchanged.

The temporary audit environment was supplied with the already-declared build,
type-stub and lint dependencies needed to run all checks, including `types-PyYAML`
and pylint. Updating its old setuptools also allowed the editable install to
produce real project metadata instead of an unsupported-backend `UNKNOWN` result.
The temporary `UNKNOWN` install was removed and the actual package was installed.
No model weights or datasets were downloaded for these checks. Tool versions are
recorded in [environment_f10_f14.json](results/environment_f10_f14.json).

The checks use the repository's actual CI scopes/options: Black/isort over
`sparse_attention_hub`, `tests` and `scripts`; flake8 over the same directories;
mypy over the full core package; and pylint with `.pylintrc` and its 8.0 threshold.
The detailed final results are below.

### F14 — State the runtime contract and verify real documentation commands

[pyproject.toml](../../pyproject.toml) now requires Python 3.10+, removes the 3.8/3.9
classifiers, includes 3.12, and pins Transformers to the verified **4.57.1** registry
contract instead of advertising support back to 4.20. Black's grammar targets no
longer include unsupported Python 3.8/3.9 source. The reference CPU checks use
PyTorch 2.8.0; this is not certification of all Torch/Python combinations allowed
by dependency resolution.

The [main README](../../README.md) and [quantization guide](../../sparse_attention_hub/kv_quantization/README.md)
describe the runtime, actual flat test-file locations, implemented/pending methods,
diagnostic controls and deferred F09 status. Their quantization verification
commands point to existing offline tests rather than a nonexistent demo or test
subdirectories:

```bash
python -m pytest -q -m unit tests/unit/kv_quantization
python -m pytest -q -m integration tests/integration/test_kv_quantization_flow.py tests/integration/test_kv_quantization_sliding.py
```

[CI](../../.github/workflows/ci.yml) now contains a focused CPU compatibility job for
Python 3.10 and 3.12. It installs PyTorch 2.8.0 CPU plus the project, checks installed
metadata/marker behavior, and runs those documentation commands with HF offline.
Only Python 3.10 was available locally; the new remote matrix has not been executed
in this workspace. [test_supported_runtime.py](../../tests/unit/benchmark/test_supported_runtime.py)
checks the installed package's Python/dependency metadata and imports the exact
HF cache/attention/mask registry entry points used by the adapter.

### Verification record

These results use Python 3.10.12, PyTorch 2.8.0 and Transformers 4.57.1 on CPU.
CUDA was unavailable. No pretrained weights or benchmark datasets were downloaded.
The new regression files add **27 cases** for worker delivery, config export,
marker collection and installed-runtime metadata. The old F11 reproduction is
now an ordinary passing test. F09 is the only remaining expected failure.

| Check | Result | Evidence |
|---|---|---|
| Quantization, all benchmark unit tests, adapter/attention tests, local-model integrations and the independent audit | **1,013 passed, 2 skipped, 1 xfailed** | [Full targeted log](results/f10_f14_tests.log) |
| Documented quantization unit category command | **503 passed**; none deselected | [Unit marker log](results/f10_f14_unit_marker.log) |
| Documented local-model integration category command | **15 passed**; none deselected | [Integration marker log](results/f10_f14_integration_marker.log) |
| Installed-runtime metadata checks | **2 passed** | [Metadata log](results/f10_f14_runtime_metadata.log) |
| Broader unit suite, excluding the separate heavy DoubleSparse correctness file | **1,417 passed, 8 skipped, 1 failed** | [Broader log](results/f10_f14_broader_unit.log) |
| Black and isort, complete CI scope plus changed benchmark/audit modules | **Passed**, 222 Python files checked | [Style log](results/f10_f14_style.log) |
| flake8, complete CI scope | **Passed**, no diagnostics | [Lint log](results/f10_f14_flake8.log) |
| mypy, complete core package with CI options | **Passed**, 110 source files | [Type log](results/f10_f14_mypy.log) |
| pylint, complete core package | **Passed**, **9.09/10** against the 8.0 threshold | [Pylint log](results/f10_f14_pylint.log) |

The rows overlap and should not be added together. The two targeted skips are
existing benchmark tests. `xfail` means a known failing assertion, not a passing
correctness test; here it is the deliberately deferred F09 extension.

The broader failure remains
`TestAdaptiveSamplingMasker.test_get_std_estimate_using_base_sample` in
[test_adaptive_sampling.py](../../tests/unit/sparse_attention/research_attention/maskers/sampling/test_adaptive_sampling.py).
The original audit and the F08 run recorded the same failure. Repeated sampled
indices are represented once in the mask, so the test's sampled standard deviation
can differ from the mask's result. Its code is unchanged by these fixes. This
separate sparse-sampling issue remains open; the entire repository is not green.
The heavy DoubleSparse file was tested separately during the original audit
(81 passed), and was not rerun for F10–F14.

To reproduce the targeted run from the repository root, with the project's dev
dependencies installed:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m pytest -q \
  tests/unit/kv_quantization tests/unit/benchmark \
  tests/integration/test_kv_quantization_flow.py \
  tests/integration/test_kv_quantization_sliding.py \
  tests/unit/adapters \
  tests/unit/sparse_attention/test_base.py \
  tests/unit/sparse_attention/test_configs_and_factories.py \
  tests/unit/sparse_attention/research_attention/test_base.py \
  verification/kv_quantization
```

The recorded local runs used `/tmp/skylight-audit-venv/bin/python`, disabled
automatic third-party pytest plugins with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and
explicitly loaded the project's required mock plugin with `-p pytest_mock`.
`MPLCONFIGDIR=/tmp/skylight-audit-mpl` kept plot-cache writes in the temporary area.

Remaining work is **F03 paper/reference reproduction**, **F09 if a method actually
requires post-attention work**, and the previously agreed later work on cache-owned
mask policies and combined sparse/quantized benchmarking. The unrelated adaptive
sampling failure also needs separate investigation. The added Python 3.10/3.12 CI
matrix has not been run remotely here. These CPU checks do not establish GPU
kernel parity, end-to-end quality, total inference-memory savings or speedups.

## 31 October 2026 — Verification before implementing paper recipes

This entry records a review, not another production fix. The complete
[preflight report](preflight/README.md) and its scripts check the current working
tree from benchmark worker through model/cache execution and result files.

- Added **344 checks** spanning component combinations, real CPU benchmark-worker
  runs, independent format math, lossless model controls and scaffold selection.
- The combined regression run has **1,357 passed, 2 skipped, 1 expected failure**
  ([log](results/preflight_final_tests.log)). The expected failure remains F09.
- Pinned official KIVI, Kitty and Microsoft MX source revisions and recorded
  numerical comparisons. The expected matching subsets pass. Default low-precision
  KIVI numerics and Kitty's value buffer/page selection have reference differences
  that still need recipe-specific work; they have not been silently fixed here.
- The lossless stored-cache control matches the ResearchAttention OFF path exactly
  for the tested full-attention models. Native HF uses different attention
  arithmetic; use a matched backend for quantization-only baselines.

The decision is to proceed with implementing/validating KIVI on the existing core,
then address each paper's specific requirements. **F03 and F09 remain deferred**
as previously agreed. Exact KVQuant and TurboQuant-QJL need additional interfaces;
the report identifies them. Production implementation files were unchanged by
this verification turn.
