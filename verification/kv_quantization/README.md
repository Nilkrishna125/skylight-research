# KV quantization verification — 30 October 2026

**Latest readiness review:** [Preflight report](preflight/README.md), 31 October:
**1,357 passed, 2 skipped, 1 expected failure**, with 344 additional cases and
pinned official-source comparisons. The shared core is ready for method
development within the tested scope. KIVI numerical parity and Kitty page/buffer
differences are explicitly recorded; pending recipes are not certified.

**Fix status, 31 October:** F01, F02, F04 and F05–F08 are implemented. F10–F14 now
cover worker diagnostics, stable configuration export, test-category markers,
quality checks and the documented runtime/CI contract. **F03 and F09 are deferred
by agreement:** full KIVI reference parity remains future work, and no current
production policy needs the disconnected after-attention hook. F03's flush-boundary
regression already passes after F01.

See [FIXES.md](FIXES.md) for plain-language explanations, source locations, snippets
and current evidence: **1,013 passed, 2 skipped, 1 expected failure (deferred F09)**
in the targeted run. The broader unit run still has the previously recorded
adaptive-sampling failure in unchanged code. All configured quality gates pass
locally. The original audit below describes commit `449d87b` before these fixes;
its findings, counts and proposed next steps are retained as historical evidence.

## Verdict

**The component framework works and has substantial useful tests. Continue developing it. Do not yet treat it as a verified reproduction of the named papers, or use the current results as evidence of total inference-memory or speed improvements.**

Two confirmed correctness defects need attention first: batched attention can lose causality, and a real HF sliding cache silently produces stale K/V. KIVI additionally needs an explicit choice between the paper pseudocode and released-reference timing; they differ. Several lifecycle, measurement, and benchmark-export issues remain despite all 230 existing quantization tests passing.

This review changes **no production implementation and no existing tests**. The new files in this directory are the report, independent checks, known-issue reproductions, and result logs. Known issues have not been fixed during the review.

## Scope and provenance

- Reviewed branch: `kv-quantization`, initially clean.
- Reviewed HEAD: `449d87b2a60c4bb8524e8cd673b5cba2da6f1d51`.
- Comparison base: `dcf24d6` (`quantization boiler`).
- Changes reviewed: `f14abea` (hook, transforms, cache policies) and `449d87b` (floating codecs, precision policies, nonuniform codec, KIVI).
- The two commits change 52 files, with 4,365 insertions and 511 deletions.
- Review covered the changed implementation and tests, their parent classes and registries, HF callback/generation/cleanup, benchmark logging/config export, documentation, and CI commands.

The old virtual environment was absent. An isolated `/tmp/skylight-audit-venv` was created with system packages visible to reuse the existing PyTorch installation. Missing dependencies were installed there; global packages were not replaced. The runtime was Python 3.10.12, PyTorch 2.8.0+cu128, Transformers 4.57.1, and pytest 8.4.2. CUDA was unavailable. Exact versions and the downloaded DoubleSparse/Quest reference commit IDs are in [environment.json](results/environment.json).

No pretrained model weights or benchmark datasets were downloaded. HF checks use tiny randomly initialized models. This is one tested dependency combination, not a compatibility certification for every version permitted by `pyproject.toml`.

## Results actually obtained

| Check | Result | Evidence |
|---|---|---|
| Existing quantization unit tests plus Qwen3 adapter integration | **230 passed**: 227 unit cases and 3 integration cases | [Log](results/existing_quantization_tests.log) |
| Independent arithmetic, packing, dtype/layout, ownership and lifecycle checks | **61 passed** | [Audit tests](test_audit_regressions.py), [log](results/audit_regressions.log) |
| Reproductions of unresolved contracts/defects | **11 strict expected failures**, representing ten findings; F08 has two codec cases | [Unmasked failures](results/audit_failures_unmasked.log) |
| Additional HF checks | **16 passed**: 3 architectures × 5 codecs, plus adapter failure cleanup | [Tests](test_hf_model_matrix.py), [log](results/hf_model_matrix.log) |
| Broader unit suite, excluding the separate heavy DoubleSparse correctness file | **1,114 passed, 8 skipped, 1 failed** | [Log](results/full_unit_tests_with_references.log) |
| Separate DoubleSparse correctness sweep | **81 passed**; reported separately from the preceding total | [Log](results/upstream_doublesparse_all.log) |
| Existing quantization tests: statement coverage | **1,416 / 1,494 = 94.78%** | [Coverage summary](results/coverage_summary.log), [JSON](results/coverage.json) |
| Existing quantization tests: branch coverage | **253 / 304 = 83.22%** | Same coverage artifacts |
| Combined statement/branch coverage | **92.83%** | This is the approximately 93% headline shown by coverage.py |
| Targeted mypy with repository strict optional settings | **3 errors** | [Log](results/mypy.log) |
| Targeted mypy with CI's `--no-strict-optional` | **2 errors remain** | [Log](results/mypy_ci_optional_flags.log) |
| Formatting/import/lint checks on current quantization code and tests | **20 files need Black formatting; 6 fail isort; 3 flake8 diagnostics** | [Black](results/black.log), [isort](results/isort.log), [flake8](results/flake8.log) |

All unit-test files were exercised across those two unit-suite runs: **1,195 passed, 8 skipped and 1 failed** in total. The audit's combined run has **77 passed and 11 expected failures**; see [audit_all.log](results/audit_all.log). An expected failure is an unresolved issue, not a passing correctness assertion.

The broad-suite failure is `TestAdaptiveSamplingMasker.test_get_std_estimate_using_base_sample`. Its test and implementation are unchanged by these two commits. The test assumes sampling has no repeated indices; repeated samples and a deduplicated mask can produce different standard deviations. This is an unresolved failure in unchanged code, not evidence by itself that the new quantizer mathematics is wrong. The broader suite must not be described as fully green.

Initial broad-suite collection also needed the `mock` package and external DoubleSparse/Quest repositories. Those setup obstacles were resolved before the final broader run. Earlier setup-failure output is retained separately in [full_unit_tests.log](results/full_unit_tests.log).

## The current execution flow

```text
ResearchAttentionConfig(kv_quantization_config=...)
    -> SparseAttention.__init__
       -> KVQuantization.create_from_config
          -> build transforms, main/region quantizers, policies, optional metric

ModelAdapterHF.process_request, independently for each question
    -> create a fresh sparse_meta_data dictionary
    -> native HF context prefill, outside enable_sparse_mode
    -> enable_sparse_mode: register/select the custom HF attention callback
    -> consume the question, then decode generated tokens
       -> model computes K/V and updates its native HF cache
       -> callback forwards layer_idx and sparse_meta_data
       -> ResearchAttention.custom_attention
          -> SparseAttention.quantize_kv
             -> KVQuantization.process
                -> get/create the request-local QuantizedKVStore
                -> append only the unseen suffix for this layer
                -> encode sinks, move eligible local tokens into the middle
                -> transform -> precision plan -> quantize
                -> store.get: restore sink + middle + local in token order
          -> apply sparse layer gating, maskers, and attention to restored K/V
    -> finally: clear_request_state removes and clears the extra store
```

Important verified details:

- `KVQuantization` owns reusable components; histories live in the metadata's store, separately for each layer and independently for K and V.
- Registry lookup uses the exact configuration type. `KIVIConfig` now builds an actual `KIVI` coordinator; five other named-method classes still raise through `PendingRecipe`.
- The hook is a parent **helper**, not the shared public wrapper discussed earlier. `ResearchAttention` explicitly calls it. Future attention implementations must call it too. Efficient attention remains outside the adapter's supported execution path.
- Quantization happens before sparse layer gating. Excluding a layer from sparse masking does not exclude it from quantization. Comments promising an exact native fallback should be read with this qualification.
- The store clones owned incoming regions. Older middle chunks are retained; their source HF prefix is not recopied and requantized on every call.
- Quantized local regions are intentionally decoded and requantized when moving into the middle. A partial local-chunk split also requantizes the retained remainder. The latter can add codec/transform error and change chunk-global NVFP4 scaling; per-token INT tests do not establish that this is harmless for all codecs.
- `KVQuantization.after_attention()` exists but **is not called anywhere in the current attention path**.
- The native full-precision HF cache stays allocated. This is an additional-cache numerical experiment. Packed-store bytes are not total GPU-memory savings, and these CPU checks establish no speedup.

## Findings, ordered by practical priority

P1 means fix or explicitly resolve before trusting the affected benchmark/reproduction. P2 means a supported combination, lifecycle, measurement, or delivery contract needs correction. Some entries are conditional limitations or unfinished contracts rather than errors in the ordinary baseline.

Production paths starting with `kv_quantization/` or `sparse_attention/` below are relative to `sparse_attention_hub/`; line numbers refer to the reviewed HEAD.

| ID | Priority | Finding |
|---|---|---|
| F01 | P1 | Per-channel compression can make batched causal attention depend on future tokens |
| F02 | P1 | Rolling/sliding HF cache silently reuses stale K/V |
| F03 | P1 for released-reference reproduction | KIVI flush timing differs from the released code; paper pseudocode differs from that code too |
| F04 | P2 | Attention hook ignores requested prefill phase |
| F05 | P2 | Outlier stand-ins can widen a different numerical group's range |
| F06 | P2 | FAKE storage bytes omit retained outlier tensors |
| F07 | P2, conditional lifecycle | Recalibration/load changes the interpretation of live packed codes |
| F08 | P2, numerical edge | Finite FP16 input can decode to infinity |
| F09 | P2, unfinished lifecycle | Post-attention compression extension is unreachable from attention |
| F10 | P2 | Benchmark worker does not enable quantization diagnostics |
| F11 | P2 | Benchmark config export serializes Enum internals and runtime addresses |
| F12 | P2 | Category-marker commands deselect every new quantization test |
| F13 | P2 | Current changed code/tests fail configured quality gates |
| F14 | P2/documentation | Python compatibility claim and implementation documentation are inconsistent |

### F01 — Batched attention can violate causality

**Locations:** `sparse_attention/research_attention/base.py:136`, `kv_quantization/base.py:140`, `cache/store.py:109`, and `quantizers/common.py:64`.

The attention mask controls which reconstructed keys a query reads. However, a reconstructed key can already depend on a later, masked token through per-channel min/max statistics. `process()` compresses before attention and receives the complete key history for the whole query batch, without a query-length/current-position contract.

**Reproduction:** use four query tokens, a causal mask, key grouping over four tokens, and no local window. Keys have channel values `[0, 1, 2, 3]`. Change only the fourth token's key to `100`. An earlier query output changes from approximately `1.880797` to `1.5`. The same perturbation with quantization off leaves earlier outputs unchanged.

This is relevant to the adapter: non-hybrid models consume question/prefix tokens as a batch. It also affects direct custom-attention prefill when a compressed group overlaps the active query span. One-token decoding has no later query tokens and does not trigger this particular violation.

**Required change:** distinguish completed past history from current-query tokens. Do not let compression statistics from later query positions alter K/V used for earlier positions. Options include preserving the entire active-query region until the attention call finishes, or consuming it sequentially. The initial context import needs an explicit boundary as well; merely adding an `after_attention()` call to the existing before-compression path is insufficient.

**Acceptance:** future-token perturbation invariance, for batched questions/prefill and key-group boundaries, plus comparison of incremental and batched causal execution under a defined numerical policy. Reproduction: `test_batched_quantized_attention_preserves_causality`.

### F02 — Real sliding caches return stale history

**Locations:** `cache/store.py:69-72`, `sparse_attention/base.py:74`, and the unconditional quantization hook before layer gating.

New data is identified solely as `incoming_length - seen_tokens`. A sliding cache can replace its contents while keeping the same length. That is interpreted as “no new tokens.”

**Reproduction:** use an actual `transformers.DynamicCache` configured through `MistralConfig(sliding_window=4)`. After five updates the native callback history contains token values `[1,2,3,4]`, but the extra store returns `[0,1,2,3]`. This happens even with the ordinary all-FP local window: it is a cache-tracking error independent of low-bit numerical error.

The README calls this unsupported, but no fail-fast guard prevents silently wrong results. Sparse layer gating does not protect sliding layers because the quantization hook runs first.

**Required change:** until sliding support is implemented, reject sliding/rolling caches explicitly at a boundary that knows the model/layer cache type. A length-only check cannot detect equal-length replacement. Supporting it properly needs absolute positions, eviction/reordering rules, and matching attention-mask semantics.

**Acceptance:** actual HF sliding-cache/model tests that either produce correct position-aligned histories or raise a clear unsupported-mode error. Reproduction: `test_rolling_hf_cache_is_rejected_or_tracks_token_positions`.

### F03 — Define which KIVI reference is being reproduced

**Locations:** `implementations/kivi.py`, `policies/cache/implementations/sink_and_local.py:18`, and the before-attention compression in `base.py`.

The default grouping and post-step counts are consistent: keys group along tokens per channel; values group along channels per token; keys flush at R; values keep an R-token rolling window.

However, the released Llama implementation forms scores with the current full-precision key buffer before flushing it, and computes the value contribution before evicting a value. Our attention uses the already-flushed reconstruction. An R=8 test at the eighth key confirms the current-step difference. [Released KIVI implementation](https://github.com/jy-yuan/KIVI/blob/main/models/llama_kivi.py).

**Important qualification:** Appendix A Algorithm 1 of the paper puts decoding compression before attention; the current schedule follows that ordering. The paper's prefill returns the original K/V for attention. Thus this finding is a **released-reference discrepancy**, not proof that all before-attention KIVI algorithms are intrinsically invalid. [KIVI paper, Algorithm 1](https://arxiv.org/pdf/2402.02750).

**Required change:** choose and document the target, pin its revision, and compare current-step attention outputs around R−1, R, R+1 and prefill boundaries. If targeting released-code results, implement its ordering through a working finish phase. Also validate parameter-rounding conventions: this codec deliberately rounds offsets downward and scales upward before encoding, so identical names/defaults do not imply bit-exact agreement with the reference codec. [Reference encoding code](https://github.com/jy-yuan/KIVI/blob/main/quant/new_pack.py).

The existing “KIVI equals equivalent generic config” test proves composition, not independent paper/reference fidelity. The audit's expected failure explicitly targets the **released** current-step contract: `test_kivi_flush_boundary_matches_released_reference_current_step`.

### F04 — Prefill metadata does not reach the coordinator

**Location:** `sparse_attention/base.py:74-80`.

`quantize_kv()` forwards `position_ids` but not `phase`. Consequently, `KVQuantization.process()` always sees its default `decode`, even if metadata says `kv_quantization_phase='prefill'`.

The direct coordinator's prefill bypass works; the audit verifies that separately. Through the hook, the same request creates and quantizes a store despite `quantize_prefill=False`. The adapter's native context prefill is outside custom attention regardless of `quantize_prefill=True`; the two contracts need to be distinguished.

**Required change:** define/forward phase explicitly, document whether the flag controls direct callbacks, context prefill, question prefill, or each separately, and test the real adapter. Reproduction: `test_attention_hook_honors_prefill_metadata`.

### F05 — Kept-token filling is not group aware

**Locations:** `base.py:248-258`, `fill_kept_tokens()` at line 333.

The helper selects a preceding non-kept token across the entire chunk. It does not know the quantizer's grouping boundaries. For group size 4, input `[100,100,100,100 | 9999,1,2,3]`, and the fifth token kept exactly, it fills that token with `100` from the previous group. The second group's surviving values span `[1,3]`, but its new range becomes `[1,100]`.

Restoring the outlier itself works; protecting the remaining group's precision does not. The source comment promising no widened group range is false for this case.

**Required change:** make exclusion/statistics quantizer/group aware, with a defined rule for all-kept groups. Passing an exclusion mask into grouping/statistics is more general than assuming one neighbour rule fits every codec. Reproduction: `test_kept_token_does_not_widen_another_per_channel_group`.

### F06 — FAKE storage is undercounted

**Locations:** `base.py:253-266`, `cache/store.py:183-194`.

FAKE chunks retain `kept_index` and `kept_values` after constructing a full reconstructed tensor. `_chunk_bytes()` immediately returns the reconstructed tensor's bytes and ignores those still-owned allocations.

**Reproduction:** one kept 32-channel FP32 token produces a report of **2,048 bytes**, while stored tensors total **2,180 bytes**. The missing 132 bytes are 128 bytes of values plus a 4-byte index. PACKED-mode tests count these correctly; the gap is specifically FAKE.

**Required change:** either drop redundant kept-token tensors once they are baked into FAKE storage or count them. Keep the distinction between store-owned bytes, reusable codebooks/rotations, temporary reconstruction, and total device allocation. Reproduction: `test_fake_store_bytes_counts_every_owned_kept_token_tensor`.

### F07 — Live packed codes depend on mutable calibrated tables

**Locations:** `non_uniform.py:178-181`, `calibrate():195-199`, `load():216-218`.

Calibrated records retain a `(layer, kind)` table key, but no table version/snapshot. Recalibrating or loading another table on the same codec silently changes the reconstruction of old codes. A saved chunk representing values near zero can decode near 100 after calibrating on shifted samples. FAKE chunks are already reconstructed and would not change, breaking equivalence between modes under this lifecycle.

This is conditional: calibrate-once-before-inference is a reasonable operating contract. It currently needs explicit documentation and enforcement or versioning.

**Required change:** freeze tables during active requests, or version/fingerprint them and bind records to immutable versions. Validate loaded table dimensions, level count, ordering, finite values and head/layer/kind coverage as part of the artifact contract. Reproduction: `test_calibration_cannot_change_live_encoded_history`.

### F08 — Finite FP16 values can reconstruct as infinity

**Locations:** `quantizers/common.py:127`, `integer.py:216`, `fp8.py:99`.

Conservative scale rounding covers the input range in FP32, but reconstruction can exceed the maximum of the original output dtype. The final cast is unchecked.

**Reproduction:** a finite FP16 group alternating `-65504` and `65504` produces `+inf` with INT2 and infinities with FP8 under their default parameter settings.

**Required change:** define finite-output saturation or raise a clear error before casting. Test both signs, dtype extrema, zero/constant groups, tiny ranges and asymmetric/symmetric cases. This is an edge case, not evidence that ordinary Gaussian-KV tests are invalid. Two reproductions are parameterized under `test_finite_fp16_extrema_decode_finitely_or_raise`.

### F09 — The post-attention extension is not connected

**Locations:** `KVQuantization.after_attention()` at `base.py:303` and `ResearchAttention.custom_attention()`.

The coordinator method exists, but attention never calls it. A valid policy that schedules compression only after attention silently never compresses. The README acknowledges this; it remains unfinished implementation rather than a working extension point.

**Required change:** connect a successful-attention finish phase, including the gated dense branch. Do not accidentally commit state after a failed computation; request cleanup is a separate concern. Reproduction: `test_attention_calls_post_attention_compression_hook`.

### F10 — Benchmark diagnostics are disabled by the worker

**Location:** `benchmark/executor.py:245`.

The worker enables only `research_attention_density` and `research_attention_output_error`. The quantizer's registered key error, value error, and store-byte metrics are not included. Selecting `error_metric=True` therefore computes diagnostics that the ordinary executor does not record and can generate repeated logger warnings.

**Required change:** route the selected quantization metrics into worker logging; add a CPU benchmark-worker/logging test. Keep diagnostic runs separate from latency runs. At present this finding is confirmed by the logging configuration and logger's enabled-metric filter, not a GPU benchmark execution.

### F11 — Exported configuration contains Enum internals

**Locations:** `benchmark/utils/data.py:29-40`, used by `benchmark/base.py:271`.

The generic serializer inspects `__dict__` before treating string-valued Enum instances as primitive values. A quantizer's `granularity` is exported as an internal dictionary containing `_value_`, `_name_`, and a stringified enum-class dictionary, including function addresses. The default quantization config serializes to about 7.6 KB in this environment.

This does **not** crash the current export, but it damages stable experiment metadata and makes a simple value such as `per_token` hard to recover. The serializer predates these commits; the new enum-containing configs expose the problem in this integration.

**Required change:** handle `Enum` explicitly with its `.value` before generic object introspection, and test that configuration JSON is primitive, stable, and includes enough method identity to reproduce a run. Reproduction: `test_benchmark_config_exports_stable_primitive_enum_values`.

### F12 — New tests lack the documented category markers

Every new unit case is deselected by `pytest -m unit tests/unit/kv_quantization`: **227 deselected**. Likewise, `pytest -m integration tests/integration/test_kv_quantization_flow.py` deselects all three integration cases. See [unit selection](results/unit_marker_selection.log) and [integration selection](results/integration_marker_selection.log).

**Qualification:** CI's `scripts/run_tests.py --type ...` selects directories, so it does include these tests. The problem affects documented marker-based commands and any filtering built on them; it is not a claim that CI's current directory-based runs omit them.

**Required change:** add module-level/category markers and verify collection in both modes.

### F13 — Quality gates fail

The targeted checks report:

- mypy: `store.py:186` and `:194` return `Any` from a function declared to return `int`; strict optional checking additionally flags passing optional `chunk.context` into `inverse()` at `base.py:280`.
- flake8: an unused `LearnedRotation` import and two blank-line diagnostics in `test_transforms.py`.
- isort: six files, listed in its log.
- Black 22.10.0: twenty files, listed in its log. The check used the formatter API to avoid process-pool issues and did not rewrite production files.

**Required change:** fix typing and formatting/import issues, then run the configured checks before a PR. The new audit test files themselves pass the recorded flake8 check. These tooling failures are separate from numerical correctness.

### F14 — Compatibility/documentation does not describe the current state

- `pyproject.toml` allows Python 3.9, but the new `dataclasses.field(kw_only=True)` requires Python 3.10+. CI currently checks Python 3.12 only. [Python dataclasses documentation](https://docs.python.org/3/library/dataclasses.html#dataclasses.field).
- The quantization README still calls FP8, nonuniform quantizers and precision policies pending even though they are now implemented.
- It refers to an absent `tutorials/py_examples/06_kv_quantization_flow.py` and old test subdirectories. The missing demo reference was retained from the scaffold.
- Its metadata-based prefill instructions do not match the hook's forwarding, as demonstrated in F04.
- Dependency ranges such as `transformers>=4.20.0` do not describe the much newer attention/mask-registry APIs that the repository imports. No lower-bound/version matrix was verified here.

**Required change:** update the supported Python/HF contract, working/pending lists and runnable examples, and validate documentation commands in CI.

## Numerical/component assessment

### Uniform integers and mixed bits

The main asymmetric formula is `q = round((x - offset) / scale)`, with codes clamped to `[0, 2**b - 1]`, and reconstruction `q*scale + offset`. Symmetric mode uses `qmax=2**(b-1)-1` and stores shifted signed codes without an offset. Grouping and byte packing are internally consistent in the tested cases.

The audit independently checks Python integer byte packing at 1/2/4/8 bits, a known INT2 example, and 54 combinations of integer bit width, grouping, symmetric/asymmetric mode and FP32/FP16/BF16 input. Those include batch size 2, three heads, noncontiguous inputs and partial token groups. Input ownership, output dtype/device and an error bound pass. Existing tests cover all parameter-dtype combinations and mixed channel widths.

Directed parameter rounding is an intentional numerical choice, not merely a storage optimization. It widens the grid and can change codes relative to other implementations. The tiny-range test's absolute tolerance is loose relative to its 1e-9 input scale; passing it establishes finiteness rather than useful relative accuracy. Mixed symmetric widths, invalid plan tensor types/shapes and codec-record corruption need more direct negative coverage.

`PER_TOKEN` groups channels within each token. `PER_CHANNEL` groups tokens within each channel. `PER_BLOCK` enforces divisibility by head dimension, making its within-token grouping equivalent to per-token grouping at the same group size. Generic policies check minimum move length, but do not universally enforce divisibility by per-channel group size; generic partial-group padding remains possible. KIVI adds the important R-divisibility check itself.

### FP8, NVFP4 reference codec and MX

FP8's per-group scaling and byte view agree with PyTorch's selected E4M3/E5M2 conversions on tested values. NVFP4's software path has the expected two-level scaling: E2M1 values, E4M3 block scales for 16 elements, and an FP32 chunk-global scale. Its tests check exact representable values, block scaling, tie rounding and 4.5-bit payload overhead plus a global scale. This matches the documented format structure; native-kernel parity remains untested. [NVIDIA format description](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html).

MX's exponent selection `floor(log2(amax))-emax`, E8M0 storage, block size 32 and saturation are consistent with the OCP conversion rule. Saturating a block maximum such as 7 to 6 for the chosen MXFP4 scale is expected, not automatically a bug. E2M1 midpoint tests check ties-to-even. [OCP MX specification, section 6.3](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).

F08 still applies to the ordinary INT/FP8 reconstruction cast. CUDA behavior, native NVFP4 layout/kernel equivalence, exponent/subnormal extremes and hardware performance have not been certified. MXFP6 is explicitly unsupported.

### Rotations

Identity, normalized Hadamard, signed randomized Hadamard and Gaussian-QR orthogonal rotations have working forward/inverse paths. Matrix orthogonality, norm preservation, seeded layer/kind reproducibility, original dtype restoration and integration with fake/packed storage pass.

The implementation uses dense matrices and repeated device conversion from cached matrices. It is a reference implementation, not an optimized fast-Hadamard GPU path. A KV storage rotation followed by inverse restoration is a useful numerical experiment, but does not establish reproduction of a full-model rotation paper. Learned rotation and transform calibration remain pending.

### Precision policies

Fixed bits, mixed per-channel integer packing and exact-token restoration work in their tested configurations. ChannelBoost's mean-absolute-magnitude ranking is consistent with Kitty's described heuristic. This verifies one component, not the paper's complete paged storage/kernel system. [Kitty, section 3.2](https://arxiv.org/html/2511.18643v1).

The existing outlier tests show that kept tokens themselves are exact. F05 demonstrates why that does not establish that non-kept groups retain their intended precision. F06 concerns the extra retained storage. Selection depends on chunk length and statistics; batched and one-token arrivals can select different outliers. A one-token middle chunk with the default 5% cap selects none. This should be documented as policy behavior, especially for rolling per-token value chunks.

### Nonuniform codecs and calibration

Known Gaussian Lloyd–Max tables, normalization/norm restoration, seeded error trends, zero vectors, code storage and per-channel k-means table save/load are tested. The 1/2/4/8-bit Gaussian tables were also checked to be finite in this environment.

Gaussian coordinate tables are an approximation to finite-dimensional sphere marginals after random rotation; TurboQuant's paper derives a dimension-dependent distribution. MSE and inner-product variants have different contracts, and the current code does not implement QJL. Keep any Gaussian MSE variant explicitly labeled. [TurboQuant paper](https://arxiv.org/html/2504.19874v1).

Calibrated mode currently pools sample rows across heads for each head-dimension coordinate, uses unweighted k-means, and indexes tables by layer and K/V kind. It is a useful generic codec, not yet faithful KVQuant. That paper also specifies pre-RoPE keys, sensitivity-weighted nonuniform quantization and per-vector sparse outliers; the current KVQuantNUQ scaffold explicitly says post-RoPE and needs a scope decision. [KVQuant paper](https://arxiv.org/abs/2401.18079).

F07 requires freezing/versioning tables. `KVQuantization.calibrate()` itself still raises; users currently reach the codec's `calibrate()` directly. Calibration iteration/sample validation, malformed checkpoint validation, per-head requirements and calibration artifact fingerprints remain work.

## Assessment of the added tests

| Test file | Useful evidence | Important remaining limits |
|---|---|---|
| `test_attention_wiring.py` | OFF path, reconstruction reaches attention, metadata store, unchanged inputs, layer gating, normal cleanup | No causal perturbation, real rolling cache, phase forwarding or finish-phase coverage |
| `test_cache_policies.py` | Six policies, rolling counts, bounds, incompatible per-channel moves | No absolute-position tracking or after-attention lifecycle |
| `test_cache_regions.py` | Independent K/V edges, local-to-middle semantics, partial splits, fake/packed agreement | Uses per-token INT to make chunking invariant; cannot establish that for chunk-global codecs or rotations |
| `test_transforms.py` | Orthogonality, inverses, seed behavior, error improvement on synthetic outliers | No GPU/layout/performance validation; synthetic distribution is not a paper benchmark |
| `test_integer_quantizer.py` | Parameter dtype/storage, range bounds, symmetry, grouping | Narrow input-dtype/shape coverage before the audit; extreme output overflow and independent complete packing oracle missing |
| `test_fp_quantizers.py` | Format grids, exact values, ties, storage sizes, error trends | FP8 reference reuses encoder-produced scales; no native NVFP4 oracle, boundary exponent matrix or CUDA runs |
| `test_precision_policies.py` | Mixed payload sizes, selected channels/tokens, exact restoration, PACKED outlier bytes | FAKE kept-token bytes, group-crossing stand-ins, mixed symmetric paths and chunk-dependent policy behavior |
| `test_non_uniform_quantizer.py` | Known Lloyd–Max values, calibrated table behavior, checkpoint roundtrip | Mutable live table hazard, malformed checkpoints, complete calibration lifecycle and paper-specific weighting |
| `test_kivi_preset.py` | Correct factory selection, defaults, schedules, full-precision tails, variants | Generic-equivalence test shares the implementation; no independent current-step paper/reference oracle |
| `test_kv_quantization_flow.py` | Real local Qwen3 adapter, all layers, store cleanup, generated-text agreement | Text equality can hide logit differences; only one architecture/codec/config, no failure-path cleanup |

The audit adds teacher-forced, fixed-token logit checks over Llama, Qwen3 and full-attention Mistral with INT2, FP8, software NVFP4, MXFP4 and Gaussian nonuniform coding. FAKE/PACKED logits match exactly in all 15 combinations, quantization changes the OFF outputs, both layers have compressed state, and the original FP32 HF cache remains intact. A separate injected-error test confirms store cleanup and restoration of the native attention implementation after generation fails.

Coverage is useful for finding missing paths, not proof of numerical correctness. Existing tests have excellent line coverage on several components while all the audit's unresolved behaviors remain. In particular, existing tests cover only about 55% of the metric module's combined statements/branches, do not exercise coordinator `after_attention()`, and do not test several store/schema/ownership rejection paths. The audit restores some of those checks.

## Remaining implementation and recommended path

### First: make one supported execution contract reliable

1. **Fix F01 and fail fast for F02.** Work in `sparse_attention/base.py`, `research_attention/base.py`, `adapters/huggingface.py`, `kv_quantization/base.py` and `cache/store.py`. Pass sufficient query/position/cache information to identify current versus completed history. Initially support append-only full attention explicitly. Also explicitly reject training/unsupported cache reordering rather than silently relying on detached inference tensors.
2. **Connect phase and finish lifecycle (F04/F09).** Preserve request-local ownership and successful failure cleanup. Cover both the normal sparse path and gated dense returns. Define native context prefill versus question prefill versus one-token decoding.
3. **Choose a KIVI target (F03).** Pin paper/released-code semantics and parameter rounding. Add independent cache-and-attention oracles at R and group boundaries, including prompt lengths below/equal/above R, long first import and batched questions. Do not change an expected-failure marker into a weak tolerance merely to make a test green.
4. **Fix numerical/storage contracts (F05/F06/F07/F08).** Make exclusion group aware; eliminate/count redundant FAKE tensors; freeze/version codebooks; define finite output casting. Verify independently owned tensor bytes and FP16/BF16 edge cases.
5. **Repair benchmark delivery (F10/F11/F12/F13/F14).** Enable selected diagnostics, serialize enums as values, attach method/config/version identity, add markers, pass static checks, update Python/HF requirements and documentation.

Each step should include the regression that currently demonstrates its failure. Keep expected failures visibly tracked until the agreed contract is implemented, then remove the relevant xfail. The audit is outside the project's default test collection; promote accepted regressions into the normal unit/integration suite as fixes land.

### Then finish the named methods, one at a time

| Method | Actual current state | Next implementation files and acceptance |
|---|---|---|
| KIVI | Runnable preset; grouped K/V codecs and buffer counts implemented | `implementations/kivi.py`, common lifecycle/policy files: settle reference schedule and codec convention, then compare teacher-forced logits against an independent reference |
| Kitty | Buffering and magnitude-based mixed channel bits exist; **named class still raises** | `implementations/kitty.py`, `channel_boost.py`, `sink_and_local.py`, integer payload layout: validate per-page ranking, sinks/local state, bit metadata and exact intended precision ratio before enabling the class |
| QuaRotKV | Randomized Hadamard component exists; **named class still raises** | `implementations/quarot_kv.py` and transforms: define a KV-only variant explicitly, or add the full model/query/weight rotation requirements before claiming full QuaRot reproduction |
| TurboQuantKV | Random rotation and Gaussian nonuniform MSE ingredients exist; **named class still raises** | `implementations/turboquant_kv.py`, `non_uniform.py`: choose finite-dimensional versus Gaussian approximation, norm precision and supported bit widths; label MSE-only and keep QJL/IP work separate |
| KVQuantNUQ | Generic calibrated k-means codec exists; **named class still raises** | `implementations/kvquant_nuq.py`, `non_uniform.py`, model/position integration: set calibrated mode, calibration artifacts/weighting/head granularity and sparse outliers; resolve pre-RoPE versus explicitly named post-RoPE variant |
| NVFP4KV | Software NVFP4 reference codec works; **native codec and named class still raise** | `implementations/nvfp4_kv.py`, quantizer `nvfp4.py`: first enable a clearly named software experiment with independent format tests, then compare real GPU kernels bit/layout-wise on supported hardware |
| Learned rotations / generic calibration | Contracts declared, not implemented | `transforms/implementations/learned_rotation.py`, coordinator calibration API and artifact/version handling |

A sensible order is **KIVI correctness → Kitty numerical implementation → remaining paper variants**, because the first two reuse most of the current machinery. Kernel optimization and systems-memory work should be a separate milestone. The existing codebase is worth retaining; replacing the architecture wholesale is not required to fix the identified contracts.

### Before drawing benchmark conclusions

- First run small teacher-forced model comparisons using fixed tokens, fixed weights, fixed context/question boundaries and recorded revisions. Include dense OFF, quantization-only, sparse-only, and sparse+quantization configurations.
- Then run short real-model quality checks and the target paper's tasks/settings. Record whether a sample is an independent request or a continuing conversation. The current adapter treats each question independently with a fresh native prefill and extra store; it does not accumulate earlier question/answer turns.
- Log local K/V reconstruction error separately from attention-output error, logits/quality metrics, store bytes, total peak device memory, and timing. The current sparse-output error is computed using the already-quantized K/V and isolates sparsity under that representation; it is not total error versus the original full-precision model.
- Benchmark warmup, synchronization and diagnostic overhead explicitly on a GPU. Measure native HF cache plus extra store plus temporary reconstruction. Do not subtract the original cache from reported peak memory while it is still allocated.
- For true KV-memory savings and native low-bit speedups, plan either an integration that avoids retaining the original full-precision cache, or an engine/kernel path that consumes the packed representation. That is beyond the present additional-cache architecture.

## Reproduce this review

From the repository root, use a Python environment with the project's dependencies. The audit environment used here is recorded above; the temporary path is local to this machine/session.

```bash
AUDIT_PY=/tmp/skylight-audit-venv/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/skylight-audit-mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# Existing tests: expected 230 passing cases on the reviewed commit/environment.
"$AUDIT_PY" -m pytest -p pytest_mock -q \
  tests/unit/kv_quantization tests/integration/test_kv_quantization_flow.py

# Independent checks plus visibly tracked known issues.
"$AUDIT_PY" -m pytest -p pytest_mock -q \
  verification/kv_quantization/test_audit_regressions.py -rx

# Show the actual failing assertions instead of marking them expected failures.
# A nonzero exit is expected on the reviewed commit.
"$AUDIT_PY" -m pytest -p pytest_mock -q \
  verification/kv_quantization/test_audit_regressions.py --runxfail

# Actual HF architecture/codec matrix and adapter failure cleanup.
"$AUDIT_PY" -m pytest -p pytest_mock -q \
  verification/kv_quantization/test_hf_model_matrix.py

# Coverage without relying on the older system pytest-cov plugin.
export COVERAGE_FILE=/tmp/skylight-audit.coverage
"$AUDIT_PY" -m coverage run --branch \
  --source=sparse_attention_hub/kv_quantization -m pytest -p pytest_mock -q \
  tests/unit/kv_quantization tests/integration/test_kv_quantization_flow.py
"$AUDIT_PY" -m coverage report -m
```

The broader unit suite contains upstream comparison tests that bootstrap external repositories; their availability is required even though model/dataset downloads are disabled. It also contains large CPU comparison matrices. Review those requirements before interpreting a collection/setup error as an algorithm failure.

## Limits of this verification

This is a source audit plus executable CPU checks against a specific commit and dependency stack. It does not prove absence of every defect. It did not measure trained-model task quality, native NVFP4, GPU kernels, CUDA performance/memory, all supported dependency versions, multi-GPU/offload, beam reordering, speculative rollback, concurrent adapters, padded production batches, or long real-model runs. Unsupported modes need either explicit guards or their own correctness tests.

Successful fake/packed parity proves agreement between two storage representations of this implementation; it does not independently prove agreement with a paper, a hardware kernel, or the original full-precision model. The findings and remaining-work list above preserve those distinctions.
