"""Exact numerical and storage comparisons with the pinned QuaRot source."""

import ast
import importlib.resources
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    QUAROT_REFERENCE_REVISION,
    ParamDType,
    PrecisionPlan,
    QuantGranularity,
    QuaRotQuantizer,
    QuaRotQuantizerConfig,
)
from sparse_attention_hub.kv_quantization.quantizers.implementations import quarot
from tests.reference.quarot import NUMERICAL_NAMES, encode, independent_unpack

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize(
    "mode,group", [("key_head", 128), ("value_group", 128), ("value_group", 32)]
)
@pytest.mark.parametrize("clip", [0.95, 1.0])
def test_codes_parameters_and_reconstruction_match_official(
    dtype, bits, symmetric, mode, group, clip
):
    x = torch.randn(2, 3, 4, 256, generator=torch.Generator().manual_seed(824)).to(
        dtype
    )[..., ::2]
    original = x.clone()
    expected = encode(
        x, mode=mode, group_size=group, bits=bits, symmetric=symmetric, clip_ratio=clip
    )
    codec = QuaRotQuantizer(
        QuaRotQuantizerConfig(
            bits=bits,
            group_size=group,
            parameter_mode=mode,
            symmetric=symmetric,
            clip_ratio=clip,
        )
    )
    actual = codec.quantize(x)
    assert torch.equal(independent_unpack(actual), expected.logical_codes)
    assert actual.scales.dtype == expected.compact_scales.dtype
    assert torch.equal(actual.scales, expected.compact_scales)
    if symmetric:
        assert actual.zero_points is None
    else:
        assert actual.zero_points.dtype == expected.compact_zero_points.dtype
        assert torch.equal(actual.zero_points, expected.compact_zero_points)
    assert torch.equal(codec.dequantize(actual), expected.reconstructed)
    assert actual.codes.numel() == (x.numel() * bits + 7) // 8
    expected_bytes = (
        actual.codes.numel()
        + expected.compact_scales.numel()
        * expected.compact_scales.element_size()
        * (1 if symmetric else 2)
    )
    assert actual.storage_bytes() == expected_bytes
    assert actual.metadata["reference_revision"] == QUAROT_REFERENCE_REVISION
    assert torch.equal(x, original)


def test_numerical_bodies_remain_identical_to_pinned_source_and_license_is_packaged():
    directory = Path(__file__).resolve().parents[2] / "reference/quarot"
    original = ast.parse((directory / "quant_utils.py.txt").read_text())
    current = ast.parse(Path(quarot.__file__).read_text())

    class IgnoreAnnotations(ast.NodeTransformer):
        @staticmethod
        def strip_docstring(node):
            if ast.get_docstring(node) is not None:
                node.body = node.body[1:]
            return node

        def visit_arg(self, node):
            node.annotation = None
            return node

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            node.returns = None
            return self.strip_docstring(node)

        def visit_ClassDef(self, node):
            self.generic_visit(node)
            return self.strip_docstring(node)

        def visit_AnnAssign(self, node):
            assert node.value is None, "Only annotation-only class fields are ignored"
            return None

    def definitions(module):
        return {
            node.name: ast.dump(
                IgnoreAnnotations().visit(node), include_attributes=False
            )
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in NUMERICAL_NAMES
        }

    assert definitions(original) == definitions(current)
    license_file = importlib.resources.files(
        "sparse_attention_hub.kv_quantization"
    ).joinpath("LICENSE-QUAROT")
    assert license_file.read_bytes() == (directory / "LICENSE").read_bytes()


@pytest.mark.parametrize("mode", ["key_head", "value_group"])
@pytest.mark.parametrize("constant", [0.0, 3.25, -3.25])
def test_zero_and_constant_reference_behavior_is_explicit(mode, constant):
    x = torch.full((1, 2, 3, 128), constant, dtype=torch.float16)
    codec = QuaRotQuantizer(QuaRotQuantizerConfig(parameter_mode=mode))
    if mode == "value_group" and constant != 0:
        with pytest.raises(ValueError, match="NaN|infinity|nonpositive"):
            codec.quantize(x)
    else:
        assert torch.equal(codec.fake_quantize(x), encode(x, mode=mode).reconstructed)


def test_key_and_value_parameters_follow_their_distinct_upstream_dtypes():
    x = torch.linspace(-1, 2, 128).half().reshape(1, 1, 1, 128)
    key = QuaRotQuantizer(QuaRotQuantizerConfig(parameter_mode="key_head")).quantize(x)
    value = QuaRotQuantizer(QuaRotQuantizerConfig()).quantize(x)
    assert key.scales.dtype == key.zero_points.dtype == torch.float32
    assert value.scales.dtype == value.zero_points.dtype == torch.float16
    assert key.scales.numel() == value.scales.numel() == 1


def test_records_own_parameters_and_survive_another_encode_and_input_mutation():
    x = torch.randn(1, 2, 4, 128).half()
    codec = QuaRotQuantizer(QuaRotQuantizerConfig())
    record = codec.quantize(x)
    before = codec.dequantize(record).clone()
    codec.quantize(x * 20)
    x.zero_()
    assert torch.equal(codec.dequantize(record), before)
    assert (
        record.scales.untyped_storage().nbytes()
        == record.scales.numel() * record.scales.element_size()
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"bits": True},
        {"bits": 5},
        {"group_size": 0},
        {"clip_ratio": 0},
        {"clip_ratio": float("nan")},
        {"clip_ratio": float("inf")},
        {"clip_ratio": True},
        {"parameter_mode": "bad"},
        {"symmetric": 1},
        {"granularity": QuantGranularity.PER_CHANNEL},
        {"scale_dtype": ParamDType.FP32},
    ],
)
def test_invalid_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        QuaRotQuantizer(QuaRotQuantizerConfig(**settings))


def test_invalid_inputs_groups_and_unsupported_plans_are_rejected():
    codec = QuaRotQuantizer(QuaRotQuantizerConfig())
    for x in (
        torch.ones(3),
        torch.ones(1, 1, 1, 128, dtype=torch.int32),
        torch.full((1, 1, 1, 128), float("nan")),
        torch.ones(1, 1, 1, 127),
        torch.ones(1, 1, 1, 128, dtype=torch.float64),
    ):
        with pytest.raises(ValueError):
            codec.quantize(x)
    x = torch.randn(1, 1, 2, 128)
    for plan in (
        PrecisionPlan(4, keep_tokens=torch.zeros(1, 1, 2, dtype=torch.bool)),
        PrecisionPlan(4, channel_bits=torch.ones(1, 1, 128)),
    ):
        with pytest.raises(NotImplementedError):
            codec.quantize(x, plan)


@pytest.mark.parametrize(
    "change", ["format", "shape", "dtype", "metadata", "scale", "zero", "count"]
)
def test_corrupt_records_are_rejected(change):
    codec = QuaRotQuantizer(QuaRotQuantizerConfig())
    record = codec.quantize(torch.randn(1, 1, 2, 128))
    if change == "format":
        record = replace(record, format_id="wrong")
    elif change == "shape":
        record = replace(record, shape=(1, 1, 2, 127))
    elif change == "dtype":
        record = replace(record, scales=record.scales.half())
    elif change == "metadata":
        record = replace(record, metadata={**record.metadata, "code_bias": 4})
    elif change == "scale":
        record = replace(record, scales=torch.zeros_like(record.scales))
    elif change == "zero":
        record = replace(record, zero_points=record.zero_points + 0.5)
    else:
        record = replace(record, metadata={**record.metadata, "code_count": 1})
    with pytest.raises(ValueError):
        codec.dequantize(record)


def test_native_value_scale_underflow_and_changed_default_dtype_are_rejected():
    codec = QuaRotQuantizer(QuaRotQuantizerConfig())
    x = torch.zeros(1, 1, 1, 128, dtype=torch.float16)
    x[..., 0] = torch.finfo(torch.float16).smallest_normal / 1024
    with pytest.raises(ValueError):
        codec.quantize(x)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match="FP32 default"):
            codec.quantize(torch.randn(1, 1, 1, 128).float())
    finally:
        torch.set_default_dtype(previous)
