"""TurboQuant Algorithms 1/2 with original-coordinate dense reconstruction.

Reference: https://arxiv.org/pdf/2504.19874v1, Sections 3.1 and 3.2.
Independent implementation of the mathematical algorithms; no community code is
copied. FP32 is our explicit runtime numerical policy, not an author kernel claim.
"""

import hashlib
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ...cache.block import QuantizedTensor
from ...context import AttentionContext
from ...numerics import finite_cast, require_finite
from ...policies.precision.base import PrecisionPlan
from ...registry import KVQuantizerRegistry
from ...transforms.implementations.random_rotation import (
    RandomRotation,
    RandomRotationConfig,
)
from ..base import KVQuantizer, KVQuantizerConfig, ParamDType, QuantGranularity
from ..common import check_input, pack_bitstream, unpack_bitstream
from .non_uniform import lloyd_max_gaussian, nearest_level

TURBOQUANT_REFERENCE = "arxiv:2504.19874v1:algorithms-1-2"
TURBOQUANT_NUMERICS = "fp32-beta-gl96-lloyd2000-tol1e-11-qr64-qjl64-v1"
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_DISTRIBUTIONS = ("sphere_beta", "gaussian_limit")


def _check_dimension(dimension: int) -> None:
    if type(dimension) is not int or dimension < 2:
        raise ValueError("TurboQuant requires a head dimension of at least 2")


def _check_bits(bits: int) -> None:
    if type(bits) is not int or bits not in (1, 2, 3, 4):
        raise ValueError("TurboQuant supports 1, 2, 3 or 4 nominal bits")


def sphere_lloyd_max(
    dimension: int,
    bits: int,
    *,
    tolerance: float = 1e-11,
    max_iterations: int = 2000,
    quadrature_order: int = 96,
) -> torch.Tensor:
    """Solve the sphere-coordinate scalar quantizer in FP64, returning CPU levels.

    With z=sin(theta), the density times dz is proportional to
    cos(theta)**(dimension-2). Conditional log weights avoid tiny tail masses.
    """
    _check_dimension(dimension)
    _check_bits(bits)
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance <= 0
        or type(max_iterations) is not int
        or max_iterations < 1
        or type(quadrature_order) is not int
        or quadrature_order < 16
    ):
        raise ValueError("Invalid Lloyd-Max convergence or quadrature settings")
    nodes_np, weights_np = np.polynomial.legendre.leggauss(quadrature_order)
    nodes = torch.tensor(nodes_np, dtype=torch.float64)
    log_weights = torch.tensor(weights_np, dtype=torch.float64).log()
    count = 2**bits
    positions = torch.arange(count, dtype=torch.float64)
    if dimension <= 4:
        centres = -1 + (2 * positions + 1) / count
    else:
        normal = torch.distributions.Normal(0.0, 1.0)
        centres = normal.icdf((positions + 0.5) / count) / math.sqrt(dimension)
    for _ in range(max_iterations):
        edges = torch.cat(
            (
                centres.new_tensor([-1.0]),
                (centres[:-1] + centres[1:]) / 2,
                centres.new_tensor([1.0]),
            )
        )
        if not (edges[1:] > edges[:-1]).all():
            raise ValueError("Sphere codebook intervals must have positive width")
        angles = edges.asin()
        mid = (angles[:-1] + angles[1:]) / 2
        half = (angles[1:] - angles[:-1]) / 2
        theta = mid[:, None] + half[:, None] * nodes
        log_mass = log_weights + (dimension - 2) * theta.cos().log()
        updated = (torch.softmax(log_mass, dim=-1) * theta.sin()).sum(dim=-1)
        updated = (updated - updated.flip(0)) / 2
        require_finite(updated, name="Sphere codebook centres")
        difference = (updated - centres).abs().max().item()
        centres = updated
        if difference <= tolerance:
            if not (centres[1:] > centres[:-1]).all() or (centres.abs() >= 1).any():
                raise ValueError("Invalid converged sphere codebook")
            return centres
    raise RuntimeError("Sphere Lloyd-Max solver did not converge")


@lru_cache(maxsize=64)
def _codebook_values(dimension: int, bits: int, distribution: str) -> Tuple[float, ...]:
    # Cache immutable Python values, never a mutable tensor shared with callers.
    if distribution == "sphere_beta":
        levels = sphere_lloyd_max(dimension, bits).float()
    else:
        levels = lloyd_max_gaussian(bits).clone() / math.sqrt(dimension)
    require_finite(levels, name="TurboQuant codebook")
    if not (levels[1:] > levels[:-1]).all():
        raise ValueError("TurboQuant codebook must be strictly ordered")
    return tuple(levels.tolist())


def turboquant_codebook(
    dimension: int, bits: int, distribution: str = "sphere_beta"
) -> torch.Tensor:
    """Owned FP32 copy of a dimension-specific table; no sample calibration."""
    _check_dimension(dimension)
    _check_bits(bits)
    if distribution not in _DISTRIBUTIONS:
        raise ValueError("Unknown TurboQuant codebook distribution")
    return torch.tensor(
        _codebook_values(dimension, bits, distribution), dtype=torch.float32
    )


def stable_norm_and_unit(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize original vectors in FP32 without squaring their raw magnitude."""
    data = x.detach().float()
    largest = data.abs().amax(dim=-1, keepdim=True)
    divisor = torch.where(largest > 0, largest, torch.ones_like(largest))
    rho = torch.linalg.vector_norm(data / divisor, dim=-1, keepdim=True) * largest
    require_finite(rho, name="TurboQuant original norms")
    unit = data / torch.where(rho > 0, rho, torch.ones_like(rho))
    require_finite(unit, name="TurboQuant normalized vectors")
    return rho, unit


def _fingerprint(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((tuple(data.shape), data.dtype)).encode())
    digest.update(data.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TurboQuantQuantizerConfig(KVQuantizerConfig):
    """Whole-head scalar indices and optional residual QJL correction.

    bits is the total nominal budget: prod uses bits-1 coarse bits plus one sign
    bit. Norms are stored in FP32; inherited scale/offset/group/symmetric settings
    must keep unused defaults. Rotation is internal so residuals use original
    coordinates and normalization occurs before rotation.
    """

    bits: int = 3
    variant: str = "mse"
    codebook_distribution: str = "sphere_beta"
    rotation: RandomRotationConfig = field(default_factory=RandomRotationConfig)
    qjl_seed: int = 1
    parameter_dtype: str = field(default="fp32", init=False)
    reference: str = field(default=TURBOQUANT_REFERENCE, init=False)
    numerics_version: str = field(default=TURBOQUANT_NUMERICS, init=False)


@dataclass(frozen=True)
class _Resources:
    rotation: torch.Tensor
    levels: torch.Tensor
    projection: Optional[torch.Tensor]
    identities: Tuple[str, str, Optional[str]]


@KVQuantizerRegistry.register(TurboQuantQuantizerConfig)
class TurboQuantQuantizer(KVQuantizer):
    """Paper MSE/prod codecs, independent of query or attention implementation."""

    config: TurboQuantQuantizerConfig

    def __init__(self, config: TurboQuantQuantizerConfig) -> None:
        super().__init__(config)
        _check_bits(config.bits)
        if config.variant not in ("mse", "prod"):
            raise ValueError("TurboQuant variant must be mse or prod")
        if config.codebook_distribution not in _DISTRIBUTIONS:
            raise ValueError("Unknown TurboQuant codebook distribution")
        if (
            QuantGranularity(config.granularity) != QuantGranularity.PER_TOKEN
            or config.group_size != 32
            or type(config.group_size) is not int
            or config.symmetric is not False
            or config.scale_dtype != ParamDType.FP16
            or config.offset_dtype != ParamDType.FP16
        ):
            raise ValueError(
                "TurboQuant normalizes whole token/head vectors in FP32; leave "
                "group_size, symmetric, scale_dtype and offset_dtype at defaults"
            )
        if (
            type(config.rotation) is not RandomRotationConfig
            or type(config.rotation.seed) is not int
            or not 0 <= config.rotation.seed < 2**31
            or type(config.qjl_seed) is not int
            or not 0 <= config.qjl_seed < 2**63
        ):
            raise ValueError("Invalid TurboQuant rotation or QJL seed")
        self._rotation = RandomRotation(config.rotation)
        self._books: Dict[int, torch.Tensor] = {}
        self._cpu: Dict[Tuple[int, str, int], _Resources] = {}
        self._devices: Dict[Tuple[int, str, int, torch.device], _Resources] = {}

    @property
    def coarse_bits(self) -> int:
        return self.config.bits - (self.config.variant == "prod")

    @staticmethod
    def _validate_context(context: Optional[AttentionContext]) -> AttentionContext:
        if (
            context is None
            or type(context.layer_idx) is not int
            or not 0 <= context.layer_idx < 2**31
            or context.tensor_kind not in ("key", "value")
        ):
            raise ValueError("TurboQuant requires a valid layer/key-or-value context")
        return context

    def _resources(
        self, context: AttentionContext, dimension: int, device: torch.device
    ) -> _Resources:
        key = (context.layer_idx, context.tensor_kind, dimension)
        if key not in self._cpu:
            rotation = self._rotation.matrix(*key)
            require_finite(rotation, name="TurboQuant rotation")
            if dimension not in self._books:
                self._books[dimension] = (
                    turboquant_codebook(
                        dimension, self.coarse_bits, self.config.codebook_distribution
                    )
                    if self.coarse_bits
                    else torch.zeros(1, dtype=torch.float32)
                )
            levels = self._books[dimension]
            projection = None
            if self.config.variant == "prod":
                tag = f"turboquant-qjl-v1:{self.config.qjl_seed}:{key}"
                seed = int.from_bytes(
                    hashlib.sha256(tag.encode()).digest()[:8], "little"
                ) % (2**63)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                projection = torch.randn(
                    dimension, dimension, generator=generator, dtype=torch.float64
                ).float()
            identities = (
                _fingerprint(rotation),
                _fingerprint(levels),
                None if projection is None else _fingerprint(projection),
            )
            self._cpu[key] = _Resources(rotation, levels, projection, identities)
        device_key = (*key, device)
        if device_key not in self._devices:
            cpu = self._cpu[key]
            self._devices[device_key] = _Resources(
                cpu.rotation.to(device),
                cpu.levels.to(device),
                None if cpu.projection is None else cpu.projection.to(device),
                cpu.identities,
            )
        return self._devices[device_key]

    def state_for(
        self, context: AttentionContext, dimension: int
    ) -> Dict[str, torch.Tensor]:
        """Owned inspection copies, also usable as explicit reference inputs."""
        context = self._validate_context(context)
        _check_dimension(dimension)
        state = self._resources(context, dimension, torch.device("cpu"))
        result = {"rotation": state.rotation.clone(), "codebook": state.levels.clone()}
        if state.projection is not None:
            result["projection"] = state.projection.clone()
        return result

    def shared_state_bytes(self) -> int:
        """Owned matrix/table tensor allocations; distinct from request payload."""
        allocations = {}
        for state in (*self._cpu.values(), *self._devices.values()):
            for tensor in (state.rotation, state.levels, state.projection):
                if tensor is not None:
                    storage = tensor.untyped_storage()
                    allocations[(tensor.device, storage.data_ptr())] = storage.nbytes()
        return sum(allocations.values())

    def _metadata(
        self, context: AttentionContext, dimension: int, count: int, state: _Resources
    ) -> Dict[str, Any]:
        return {
            "reference": TURBOQUANT_REFERENCE,
            "numerics_version": TURBOQUANT_NUMERICS,
            "variant": self.config.variant,
            "distribution": self.config.codebook_distribution,
            "layer_idx": context.layer_idx,
            "tensor_kind": context.tensor_kind,
            "dimension": dimension,
            "rotation_seed": self.config.rotation.seed,
            "qjl_seed": self.config.qjl_seed,
            "coarse_bits": self.coarse_bits,
            "code_count": count if self.coarse_bits else 0,
            "qjl_code_count": count if self.config.variant == "prod" else 0,
            "rotation_id": state.identities[0],
            "codebook_id": state.identities[1],
            "projection_id": state.identities[2],
        }

    def quantize(
        self,
        x: torch.Tensor,
        plan: Optional[PrecisionPlan] = None,
        *,
        context: Optional[AttentionContext] = None,
    ) -> QuantizedTensor:
        check_input(x, plan)
        context = self._validate_context(context)
        if x.dtype not in _DTYPES:
            raise ValueError("TurboQuant inputs must be FP16, BF16 or FP32")
        if plan is not None and (
            type(plan.bits) is not int or plan.bits != self.config.bits
        ):
            raise ValueError("TurboQuant precision plan must match its configured bits")
        dimension = x.shape[-1]
        _check_dimension(dimension)
        state = self._resources(context, dimension, x.device)
        rho, unit = stable_norm_and_unit(x)
        if self.coarse_bits:
            indices = nearest_level(unit @ state.rotation, state.levels)
            indices = indices.masked_fill(rho == 0, 0)
            codes, _ = pack_bitstream(indices, self.coarse_bits)
            coarse = state.levels[indices] @ state.rotation.T
            coarse = coarse.masked_fill(rho == 0, 0)
        else:
            codes = torch.empty(0, dtype=torch.uint8, device=x.device)
            coarse = torch.zeros_like(unit)
        qjl_codes, gamma = None, None
        if state.projection is not None:
            residual = unit - coarse
            gamma = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
            require_finite(gamma, name="TurboQuant residual norms")
            qjl_codes, _ = pack_bitstream(residual @ state.projection.T >= 0, 1)
        return QuantizedTensor(
            codes=codes,
            scales=torch.empty(0, dtype=torch.float32, device=x.device),
            zero_points=None,
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=3,
            bits=self.config.bits,
            format_id=f"turboquant_{self.config.variant}_v1",
            group_size=dimension,
            metadata=self._metadata(context, dimension, x.numel(), state),
            norms=rho.detach().clone().contiguous(),
            qjl_codes=qjl_codes,
            residual_norms=None
            if gamma is None
            else gamma.detach().clone().contiguous(),
        )

    def _validate_record(
        self, q: QuantizedTensor
    ) -> Tuple[AttentionContext, _Resources]:
        if q.format_id != f"turboquant_{self.config.variant}_v1":
            raise ValueError(
                "TurboQuant record variant/format does not match this codec"
            )
        if (
            len(q.shape) != 4
            or any(type(n) is not int or n < 1 for n in q.shape)
            or q.shape[-1] < 2
            or type(q.bits) is not int
            or q.bits != self.config.bits
            or type(q.group_size) is not int
            or q.group_size != q.shape[-1]
            or q.axis != 3
            or q.dtype not in _DTYPES
        ):
            raise ValueError("Invalid TurboQuant stored layout")
        if q.zero_points is not None or q.scales.numel() != 0:
            raise ValueError("TurboQuant does not store scales or zero points")
        if q.scales.device != q.device or q.scales.dtype != torch.float32:
            raise ValueError("Invalid TurboQuant empty scale buffer")
        for extra in (
            q.global_scale,
            q.codebook,
            q.outlier_indices,
            q.outlier_values,
            q.group_bits,
        ):
            if extra is not None:
                raise ValueError("Unexpected TurboQuant payload field")
        context = self._validate_context(
            AttentionContext(
                q.metadata.get("layer_idx"), q.metadata.get("tensor_kind"), 0
            )
        )
        state = self._resources(context, q.shape[-1], q.device)
        expected = self._metadata(context, q.shape[-1], math.prod(q.shape), state)
        if any(
            type(q.metadata.get(k)) is not type(v) or q.metadata.get(k) != v
            for k, v in expected.items()
        ):
            raise ValueError(
                "TurboQuant record metadata/codebook/matrix fingerprint mismatch"
            )
        if (
            q.codes.device != q.device
            or q.codes.dtype != torch.uint8
            or q.codes.ndim != 1
        ):
            raise ValueError("Invalid TurboQuant index stream")
        norm_shape = (*q.shape[:-1], 1)
        parameters = [q.norms]
        if self.config.variant == "prod":
            parameters.append(q.residual_norms)
            if q.qjl_codes is None or q.qjl_codes.device != q.device:
                raise ValueError("TurboQuant prod requires residual sign codes")
        elif q.residual_norms is not None or q.qjl_codes is not None:
            raise ValueError("TurboQuant MSE cannot contain a QJL payload")
        for parameter in parameters:
            if (
                parameter is None
                or parameter.dtype != torch.float32
                or parameter.device != q.device
                or tuple(parameter.shape) != norm_shape
            ):
                raise ValueError("Invalid TurboQuant norm shape/dtype/device")
            require_finite(parameter, name="TurboQuant stored norms")
            if (parameter < 0).any():
                raise ValueError("TurboQuant norms must be nonnegative")
        if (
            q.residual_norms is not None
            and ((q.norms == 0) & (q.residual_norms != 0)).any()
        ):
            raise ValueError("Zero original vectors must have zero residual norms")
        return context, state

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        _, state = self._validate_record(q)
        if self.coarse_bits:
            indices = unpack_bitstream(
                q.codes, self.coarse_bits, q.metadata["code_count"]
            )
            coarse = state.levels[indices.long().reshape(q.shape)] @ state.rotation.T
            coarse = coarse.masked_fill(q.norms == 0, 0)
        else:
            if q.codes.numel():
                raise ValueError("Zero-bit coarse stage requires an empty index stream")
            coarse = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        if state.projection is not None:
            signs = unpack_bitstream(
                q.qjl_codes, 1, q.metadata["qjl_code_count"]
            ).float()
            signs = (2 * signs - 1).reshape(q.shape)
            correction = (
                (signs @ state.projection)
                * q.residual_norms
                * (math.sqrt(math.pi / 2) / q.shape[-1])
            )
            coarse = coarse + correction
        return finite_cast(coarse * q.norms, q.dtype)
