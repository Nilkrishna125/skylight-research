"""QuaRot accuracy-path numerics with Skylight grouped packed storage.

Numerical functions and ActQuantizer are adapted from spcl/QuaRot,
fake_quant/quant_utils.py, revision 5008669b08c1f11f9b64d52d16fddd47ca754c5a.
Licensed under Apache-2.0; see ../../LICENSE-QUAROT.
Changes: type annotations, torch-only dependencies and a no-op cleanup shim, then
our config, grouping/storage adapter, validation and inverse-free codec boundary.
The authors' numerical bodies are retained; the transform restores coordinates.
"""

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import torch

from ...cache.block import QuantizedTensor
from ...context import AttentionContext
from ...numerics import finite_cast, require_finite
from ...policies.precision.base import PrecisionPlan
from ...registry import KVQuantizerRegistry
from ..base import KVQuantizer, KVQuantizerConfig, ParamDType, QuantGranularity
from ..common import check_input, pack_bitstream, unpack_bitstream

QUAROT_REFERENCE_REVISION = "5008669b08c1f11f9b64d52d16fddd47ca754c5a"
_SUPPORTED_BITS = (2, 3, 4, 8)
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Upstream grouped find_params invokes only this administrative cleanup function.
# Keep its numerical body intact without importing the authors' CLI dependencies.
utils = SimpleNamespace(cleanup_memory=lambda **kwargs: None)


def get_minq_maxq(bits: int, sym: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq


def asym_quant(
    x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, maxq: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(
    q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor
) -> torch.Tensor:
    return scale * (q - zero)


def asym_quant_dequant(
    x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, maxq: torch.Tensor
) -> torch.Tensor:
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(
    x: torch.Tensor, scale: torch.Tensor, maxq: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return scale * q


def sym_quant_dequant(
    x: torch.Tensor, scale: torch.Tensor, maxq: torch.Tensor
) -> torch.Tensor:
    return sym_dequant(*sym_quant(x, scale, maxq))


class ActQuantizer(torch.nn.Module):

    """
    A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
    for the activations.
    """

    # Upstream register_buffer creates these dynamically; free() can release them.
    maxq: torch.Tensor
    scale: Any
    zero: Any

    def __init__(self) -> None:
        super(ActQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))
        self.bits = 16

    def free(self) -> None:
        self.zero = None
        self.scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        elif self.sym:
            return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
        return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(
        self, bits: int, groupsize: int = -1, sym: bool = False, clip_ratio: float = 1.0
    ) -> None:
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert (
            self.clip_ratio <= 1 and self.clip_ratio > 0
        ), "Clip ratio should be in (0, 1]"

    def find_params_per_token_groupwise(self, x: torch.Tensor) -> None:
        init_shape = x.shape
        reshaped_x = x.reshape(
            -1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize
        )

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x: torch.Tensor) -> None:
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            utils.cleanup_memory(verbos=False)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = (
                self.scale.unsqueeze(1)
                .repeat(1, reshaped_x.shape[-1])
                .reshape(init_shape)
            )
            self.zero = (
                self.zero.unsqueeze(1)
                .repeat(1, reshaped_x.shape[-1])
                .reshape(init_shape)
            )


@dataclass(frozen=True)
class QuaRotQuantizerConfig(KVQuantizerConfig):
    """Official accuracy-path parameters; group channels inside each token/head.

    key_head reproduces QKRotationWrapper's flattened-head call (groupsize=-1).
    value_group reproduces the v_proj grouped output quantizer. Parameter dtypes
    follow those distinct upstream paths. Inherited scale_dtype/offset_dtype
    remain unused defaults, as in KIVIQuantizerConfig.
    """

    bits: int = 4
    group_size: int = 128
    symmetric: bool = False
    granularity: QuantGranularity = QuantGranularity.PER_TOKEN
    clip_ratio: float = 0.95
    parameter_mode: str = "value_group"
    parameter_dtype: str = field(default="upstream_path", init=False)
    reference_revision: str = field(default=QUAROT_REFERENCE_REVISION, init=False)


@KVQuantizerRegistry.register(QuaRotQuantizerConfig)
class QuaRotQuantizer(KVQuantizer):
    """Delegate numerical calculations to the pinned authors' routines.

    Packed parameters are compact owned copies, rather than upstream's repeated
    full-shape buffers. Logical codes use our contiguous unsigned bitstream.
    """

    config: QuaRotQuantizerConfig

    def __init__(self, config: QuaRotQuantizerConfig) -> None:
        super().__init__(config)
        self._validate_bits(config.bits)
        if type(config.group_size) is not int or config.group_size < 1:
            raise ValueError("QuaRot group_size must be a positive integer")
        if QuantGranularity(config.granularity) != QuantGranularity.PER_TOKEN:
            raise ValueError("QuaRot groups channels per token (PER_TOKEN)")
        if type(config.symmetric) is not bool:
            raise ValueError("QuaRot symmetric must be boolean")
        if (
            isinstance(config.clip_ratio, bool)
            or not isinstance(config.clip_ratio, (int, float))
            or not math.isfinite(config.clip_ratio)
            or not 0 < config.clip_ratio <= 1
        ):
            raise ValueError("QuaRot clip_ratio must be finite and in (0, 1]")
        if config.parameter_mode not in ("key_head", "value_group"):
            raise ValueError("QuaRot parameter_mode must be key_head or value_group")
        if (
            config.scale_dtype != ParamDType.FP16
            or config.offset_dtype != ParamDType.FP16
        ):
            raise ValueError(
                "QuaRot parameters follow the upstream path; leave scale_dtype "
                "and offset_dtype at defaults"
            )

    @staticmethod
    def _validate_bits(bits: int) -> None:
        if type(bits) is not int or bits not in _SUPPORTED_BITS:
            raise ValueError("QuaRot supports 2, 3, 4 or 8 bits")

    @staticmethod
    def _compact(parameter: torch.Tensor, shape: tuple, group: int) -> torch.Tensor:
        return (
            parameter.reshape(*shape[:-1], shape[-1] // group, group)[..., :1]
            .detach()
            .clone()
            .contiguous()
        )

    def quantize(
        self,
        x: torch.Tensor,
        plan: Optional[PrecisionPlan] = None,
        *,
        context: Optional[AttentionContext] = None,
    ) -> QuantizedTensor:
        check_input(x, plan)
        if x.dtype not in _DTYPES.values():
            raise ValueError("QuaRot inputs must be FP16, BF16 or FP32")
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("QuaRot reference requires torch's FP32 default dtype")
        bits = self.config.bits if plan is None else plan.bits
        self._validate_bits(bits)
        group = self.config.group_size
        key_mode = self.config.parameter_mode == "key_head"
        if x.shape[-1] % group or (key_mode and group != x.shape[-1]):
            raise ValueError(
                "QuaRot requires complete groups; key_head group_size must equal "
                "head_dim and value_group group_size must divide head_dim"
            )
        # Match the upstream invocation, not just its algebraic quantization formula.
        source = x.detach()
        source = source.reshape(-1, source.shape[-1]) if key_mode else source
        reference = ActQuantizer()
        reference.configure(
            bits=bits,
            groupsize=-1 if key_mode else group,
            sym=self.config.symmetric,
            clip_ratio=self.config.clip_ratio,
        )
        reference.find_params(source)
        require_finite(reference.scale, name="QuaRot reference scales")
        require_finite(reference.zero, name="QuaRot reference zero points")
        if (reference.scale <= 0).any():
            raise ValueError("QuaRot reference produced a nonpositive scale")
        encoded = reference.quantize(source)
        logical, scales = encoded[:2]
        require_finite(logical, name="QuaRot reference codes")
        bias = 2 ** (bits - 1) if self.config.symmetric else 0
        packed, count = pack_bitstream(logical + bias, bits)
        zeros = (
            None if self.config.symmetric else self._compact(encoded[2], x.shape, group)
        )
        return QuantizedTensor(
            codes=packed,
            scales=self._compact(scales, x.shape, group),
            zero_points=zeros,
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=3,
            bits=bits,
            format_id="quarot_accuracy_int_v1",
            group_size=group,
            metadata={
                "code_count": count,
                "symmetric": self.config.symmetric,
                "code_bias": bias,
                "arithmetic_dtype": str(logical.dtype).removeprefix("torch."),
                "parameter_mode": self.config.parameter_mode,
                "clip_ratio": self.config.clip_ratio,
                "reference_revision": QUAROT_REFERENCE_REVISION,
            },
        )

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        if q.format_id != "quarot_accuracy_int_v1":
            raise ValueError(f"Unsupported QuaRot encoding: {q.format_id}")
        self._validate_bits(q.bits)
        if (
            len(q.shape) != 4
            or any(type(n) is not int or n < 1 for n in q.shape)
            or type(q.group_size) is not int
            or q.group_size < 1
            or q.shape[-1] % q.group_size
            or q.axis != 3
            or q.dtype not in _DTYPES.values()
        ):
            raise ValueError("Invalid QuaRot stored layout")
        mode = q.metadata.get("parameter_mode")
        if mode not in ("key_head", "value_group") or (
            mode == "key_head" and q.group_size != q.shape[-1]
        ):
            raise ValueError("Invalid QuaRot stored parameter mode")
        symmetric = q.metadata.get("symmetric")
        bias = 2 ** (q.bits - 1) if symmetric else 0
        if (
            type(symmetric) is not bool
            or type(q.metadata.get("code_bias")) is not int
            or q.metadata["code_bias"] != bias
            or type(q.metadata.get("code_count")) is not int
            or q.metadata["code_count"] != math.prod(q.shape)
            or q.metadata.get("reference_revision") != QUAROT_REFERENCE_REVISION
        ):
            raise ValueError("Invalid QuaRot stored metadata")
        arithmetic_dtype = torch.float32 if mode == "key_head" else q.dtype
        if q.metadata.get("arithmetic_dtype") != str(arithmetic_dtype).removeprefix(
            "torch."
        ):
            raise ValueError("Invalid QuaRot arithmetic dtype")
        expected_shape = (*q.shape[:-1], q.shape[-1] // q.group_size, 1)
        parameters = [q.scales]
        if symmetric:
            if q.zero_points is not None:
                raise ValueError(
                    "Symmetric QuaRot records must not contain zero points"
                )
        else:
            if q.zero_points is None:
                raise ValueError("Asymmetric QuaRot requires integer zero points")
            parameters.append(q.zero_points)
        for parameter in parameters:
            if (
                tuple(parameter.shape) != expected_shape
                or parameter.dtype != arithmetic_dtype
                or parameter.device != q.device
            ):
                raise ValueError("Invalid QuaRot stored parameter shape/dtype/device")
            require_finite(parameter, name="QuaRot stored parameters")
        if (q.scales <= 0).any():
            raise ValueError("QuaRot stored scales must be positive")
        if q.zero_points is not None and (q.zero_points != q.zero_points.round()).any():
            raise ValueError("QuaRot zero points must be integers")
        if q.codes.device != q.device:
            raise ValueError("QuaRot code device differs from its record")
        codes = unpack_bitstream(q.codes, q.bits, q.metadata["code_count"])
        groups = codes.to(arithmetic_dtype).reshape(
            *q.shape[:-1], q.shape[-1] // q.group_size, q.group_size
        )
        if symmetric:
            restored = sym_dequant(groups - bias, q.scales)
        else:
            restored = asym_dequant(groups, q.scales, q.zero_points)
        return finite_cast(restored.reshape(q.shape), q.dtype)
