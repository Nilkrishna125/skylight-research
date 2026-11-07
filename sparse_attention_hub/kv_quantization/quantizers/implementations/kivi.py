"""KIVI's released PyTorch encoding and unpacking arithmetic.

Adapted from jy-yuan/KIVI, revision 876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6,
quant/new_pack.py. Copyright (c) 2024 jiayi yuan, MIT; see ../../LICENSE-KIVI.
The authors' CUDA fused matrix multiplication is a separate execution backend.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

from ...cache.block import QuantizedTensor
from ...context import AttentionContext
from ...numerics import finite_cast, require_finite
from ...policies.precision.base import PrecisionPlan
from ...registry import KVQuantizerRegistry
from ..base import KVQuantizer, KVQuantizerConfig, ParamDType, QuantGranularity
from ..common import (
    check_input,
    group_for_quantization,
    group_min_max,
    layout_from_metadata,
    layout_metadata,
    pack_codes,
    ungroup_values,
    unpack_codes,
)

KIVI_REFERENCE_REVISION = "876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6"
NATIVE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True)
class KIVIQuantizerConfig(KVQuantizerConfig):
    """Asymmetric grouped INT using the released KIVI operation order.

    Minima, ranges, division and rounding use the input dtype. Parameters also
    retain that dtype; ``parameter_dtype='input'`` records this in exported config.
    Inherited scale_dtype/offset_dtype must keep their defaults and are unused.
    Use IntQuantizerConfig for experiments with independent parameter dtypes.

    Groups must be complete. KIVI's cache buffer provides complete key groups;
    the value head dimension must be divisible by its group size. Exact constants
    use scale zero/code zero, avoiding the reference's undefined 0/0 case.
    """

    bits: int = 2
    parameter_dtype: str = field(default="input", init=False)
    reference_revision: str = field(default=KIVI_REFERENCE_REVISION, init=False)


@KVQuantizerRegistry.register(KIVIQuantizerConfig)
class KIVIQuantizer(KVQuantizer):
    """Portable numerical implementation; packed bytes use Skylight's layout.

    Decoding follows the official unpack helpers: integer codes first become
    FP16, then multiply/add use PyTorch dtype promotion. The result returns to
    the incoming tensor's dtype. BF16 therefore is a portable extension; the
    authors' fused CUDA path requires FP16. Keep-token exclusion is an optional
    framework extension, not part of the unmodified KIVI recipe.
    """

    config: KIVIQuantizerConfig

    def __init__(self, config: KIVIQuantizerConfig) -> None:
        super().__init__(config)
        if type(config.bits) is not int or config.bits not in (2, 4, 8):
            raise ValueError("KIVI quantization supports 2, 4 or 8 bits")
        if type(config.group_size) is not int or config.group_size < 1:
            raise ValueError("KIVI group_size must be a positive integer")
        if config.symmetric:
            raise ValueError("KIVI reference quantization is asymmetric")
        if (
            config.scale_dtype != ParamDType.FP16
            or config.offset_dtype != ParamDType.FP16
        ):
            raise ValueError(
                "KIVI parameters follow the input dtype; leave scale_dtype and "
                "offset_dtype at defaults, or use IntQuantizerConfig for experiments"
            )
        self.granularity = QuantGranularity(config.granularity)

    def quantize(
        self,
        x: torch.Tensor,
        plan: Optional[PrecisionPlan] = None,
        *,
        context: Optional[AttentionContext] = None,
    ) -> QuantizedTensor:
        check_input(x, plan, allow_keep_tokens=True)
        if x.dtype not in NATIVE_DTYPES:
            raise ValueError("KIVI reference inputs must be FP16, BF16 or FP32")
        bits = self.config.bits if plan is None else plan.bits
        if type(bits) is not int or bits not in (2, 4, 8):
            raise ValueError("KIVI precision plan must use 2, 4 or 8 bits")
        axis = 2 if self.granularity == QuantGranularity.PER_CHANNEL else 3
        if x.shape[axis] % self.config.group_size:
            raise ValueError(
                "KIVI requires complete groups; group_size must divide the grouped dimension"
            )
        groups, layout, valid = group_for_quantization(
            x, self.granularity, self.config.group_size, plan
        )
        # Shared grouping is lossless for our three supported input dtypes. Put
        # data back in that dtype BEFORE every reference arithmetic operation.
        groups = groups.to(x.dtype)
        minimum, maximum = group_min_max(groups, valid)
        scale = (maximum - minimum) / (2**bits - 1)
        require_finite(scale, name="KIVI reference scale")
        if ((scale == 0) & (maximum != minimum)).any():
            raise ValueError(
                "KIVI reference scale underflow; use a higher input precision or the generic INT codec"
            )
        if valid is not None:
            groups = torch.where(valid, groups, minimum)
        denominator = torch.where(scale == 0, torch.ones_like(scale), scale)
        codes = (groups - minimum) / denominator
        codes = codes.clamp(0, 2**bits - 1).round()
        require_finite(codes, name="KIVI reference codes")
        packed, count = pack_codes(codes, bits)
        return QuantizedTensor(
            codes=packed,
            scales=scale.contiguous(),
            zero_points=minimum.contiguous(),
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=layout.axis,
            bits=bits,
            format_id="kivi_int_v1",
            group_size=self.config.group_size,
            metadata={
                **layout_metadata(layout, count),
                "reference_revision": KIVI_REFERENCE_REVISION,
            },
        )

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        if q.format_id != "kivi_int_v1":
            raise ValueError(f"Unsupported KIVI encoding: {q.format_id}")
        if q.zero_points is None:
            raise ValueError("KIVI encoding requires group minima")
        require_finite(q.scales, name="KIVI stored scales")
        require_finite(q.zero_points, name="KIVI stored minima")
        layout = layout_from_metadata(q.metadata, q.axis)
        codes = unpack_codes(q.codes, q.bits, q.metadata["code_count"])
        groups = codes.to(torch.float16).reshape(
            *layout.layout_shape[:-1],
            layout.padded_length // q.group_size,
            q.group_size,
        )
        restored = groups * q.scales + q.zero_points
        if not torch.isfinite(restored).all():
            # The public reconstruction rule saturates finite overflow. A native
            # half intermediate may overflow even when its wide result is finite.
            wide = groups.float() * q.scales.float() + q.zero_points.float()
            restored = torch.where(torch.isfinite(restored), restored.float(), wide)
        return finite_cast(
            ungroup_values(restored, layout, q.shape), q.dtype, device=q.device
        )
