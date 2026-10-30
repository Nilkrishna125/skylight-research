"""NonUniformQuantizer: values snap to a table of levels instead of an even grid."""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Mapping, Optional, Tuple

import torch

from ...cache.block import QuantizedTensor
from ...context import AttentionContext
from ...policies.precision.base import PrecisionPlan
from ...registry import KVQuantizerRegistry
from ..base import KVQuantizer, KVQuantizerConfig, QuantGranularity
from ..common import check_input, pack_codes, store_param, unpack_codes

MODES = ("gaussian", "calibrated")
TableKey = Tuple[int, str]  # (layer_idx, "key" | "value")


@dataclass(frozen=True)
class NonUniformQuantizerConfig(KVQuantizerConfig):
    """Snap each value to the nearest entry of a table of ``2**bits`` levels.

    Attributes:
        mode: ``"gaussian"``: every token vector is divided by its L2 norm and scaled
            by sqrt(head_dim), so its coordinates are close to N(0, 1) (especially
            after a random rotation, as in TurboQuant), then snapped to the
            Lloyd-Max levels of N(0, 1); the norm is stored in ``scale_dtype``.
            ``"calibrated"``: one table per channel, layer and keys/values, learned
            by 1-D k-means from samples in ``calibrate()`` (KVQuant-style); nothing
            is stored per token besides the codes.
        codebook_path: Load calibrated tables from this file (``save`` writes it).
        calibration_iters: k-means iterations in ``calibrate()``.

    ``bits`` must be 1, 2, 4 or 8. ``group_size``, ``symmetric`` and
    ``offset_dtype`` are unused. ``"gaussian"`` needs PER_TOKEN granularity.
    """

    bits: int = 2
    mode: str = "gaussian"
    codebook_path: Optional[str] = None
    calibration_iters: int = 50


@lru_cache(maxsize=None)
def lloyd_max_gaussian(bits: int, iterations: int = 300) -> torch.Tensor:
    """Levels that minimise the mean squared error of quantizing N(0, 1), ascending."""
    count = 2**bits
    normal = torch.distributions.Normal(0.0, 1.0)
    probs = (torch.arange(count, dtype=torch.float64) + 0.5) / count
    levels = normal.icdf(probs)
    for _ in range(iterations):
        edges = torch.cat(
            [
                torch.tensor([-math.inf], dtype=torch.float64),
                (levels[1:] + levels[:-1]) / 2,
                torch.tensor([math.inf], dtype=torch.float64),
            ]
        )
        density = torch.exp(-(edges**2) / 2) / math.sqrt(2 * math.pi)
        density = torch.nan_to_num(density, nan=0.0)
        mass = normal.cdf(edges[1:]) - normal.cdf(edges[:-1])
        levels = (density[:-1] - density[1:]) / mass  # E[X | edge_i < X < edge_i+1]
    return levels.float()


def nearest_level(values: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Index of the nearest of the ascending ``levels`` (last axis), broadcasting."""
    midpoints = (levels[..., 1:] + levels[..., :-1]) / 2
    if midpoints.ndim == 1:
        return torch.bucketize(values.contiguous(), midpoints)
    # One table per channel: compare against each channel's midpoints.
    return (values[..., None] > midpoints).sum(dim=-1)


@KVQuantizerRegistry.register(NonUniformQuantizerConfig)
class NonUniformQuantizer(KVQuantizer):
    config: NonUniformQuantizerConfig

    def __init__(self, config: NonUniformQuantizerConfig) -> None:
        super().__init__(config)
        if config.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if config.bits not in (1, 2, 4, 8):
            raise ValueError("NonUniformQuantizer supports 1, 2, 4 or 8 bits")
        if (
            config.mode == "gaussian"
            and QuantGranularity(config.granularity) != QuantGranularity.PER_TOKEN
        ):
            raise ValueError("gaussian mode normalises whole tokens; use PER_TOKEN")
        self.tables: Dict[TableKey, torch.Tensor] = {}
        if config.codebook_path is not None:
            self.load(config.codebook_path)

    # ------------------------------------------------------------------ encoding
    def quantize(
        self,
        x: torch.Tensor,
        plan: Optional[PrecisionPlan] = None,
        *,
        context: Optional[AttentionContext] = None,
    ) -> QuantizedTensor:
        check_input(x, plan)
        if plan is not None and plan.bits != self.config.bits:
            raise ValueError(f"NonUniform precision plan must use {self.config.bits} bits")
        if self.config.mode == "gaussian":
            return self._quantize_gaussian(x)
        return self._quantize_calibrated(x, context)

    def _quantize_gaussian(self, x: torch.Tensor) -> QuantizedTensor:
        dim = x.shape[-1]
        data = x.detach().float()
        exact = data.norm(dim=-1, keepdim=True)
        # Rounded up to the storage dtype; a zero vector keeps a zero norm.
        norms = torch.where(
            exact > 0, store_param(exact, self.config.scale_dtype), torch.zeros_like(exact)
        )
        unit = data / torch.where(norms > 0, norms, torch.ones_like(norms)) * math.sqrt(dim)
        levels = lloyd_max_gaussian(self.config.bits).to(x.device)
        packed, count = pack_codes(nearest_level(unit, levels), self.config.bits)
        return self._record(x, packed, count, norms, mode="gaussian", key=None)

    def _quantize_calibrated(
        self, x: torch.Tensor, context: Optional[AttentionContext]
    ) -> QuantizedTensor:
        if context is None:
            raise ValueError("calibrated mode needs the attention context (layer, kind)")
        key = (context.layer_idx, context.tensor_kind)
        if key not in self.tables:
            raise RuntimeError(
                f"NonUniformQuantizer has no table for layer {key[0]} {key[1]}s; "
                "run calibrate() or set codebook_path"
            )
        table = self.tables[key].to(x.device)  # [channels, levels]
        if table.shape[0] != x.shape[-1]:
            raise ValueError("Calibrated table does not match head_dim")
        codes = nearest_level(x.detach().float(), table)
        packed, count = pack_codes(codes, self.config.bits)
        return self._record(x, packed, count, None, mode="calibrated", key=key)

    def _record(
        self,
        x: torch.Tensor,
        packed: torch.Tensor,
        count: int,
        norms: Optional[torch.Tensor],
        *,
        mode: str,
        key: Optional[TableKey],
    ) -> QuantizedTensor:
        return QuantizedTensor(
            codes=packed,
            scales=torch.empty(0, device=x.device),
            zero_points=None,
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=3,
            bits=self.config.bits,
            format_id=f"nonuniform_{mode}_v1",
            group_size=x.shape[-1],
            metadata={"code_count": count, "table_key": key},
            norms=(
                None
                if norms is None
                else norms.to(self.config.scale_dtype.torch_dtype).contiguous()
            ),
        )

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        codes = unpack_codes(q.codes, q.bits, q.metadata["code_count"]).long()
        codes = codes.reshape(q.shape)
        if q.format_id == "nonuniform_gaussian_v1":
            assert q.norms is not None
            levels = lloyd_max_gaussian(q.bits).to(q.codes.device)
            restored = levels[codes] / math.sqrt(q.shape[-1]) * q.norms.float()
        elif q.format_id == "nonuniform_calibrated_v1":
            table = self.tables[tuple(q.metadata["table_key"])].to(q.codes.device)
            channel = torch.arange(q.shape[-1], device=q.codes.device)
            restored = table[channel, codes]
        else:
            raise ValueError(f"Unsupported non-uniform encoding: {q.format_id}")
        return restored.to(device=q.device, dtype=q.dtype).contiguous()

    # --------------------------------------------------------------- calibration
    def calibrate(self, samples: Mapping[TableKey, torch.Tensor]) -> None:
        """Learn per-channel tables for each ``(layer_idx, "key" | "value")``.

        Each value holds sample vectors with head_dim as the last axis, e.g. keys
        collected from calibration prompts. Only used in ``"calibrated"`` mode.
        """
        if self.config.mode != "calibrated":
            raise ValueError("calibrate() is for mode='calibrated'")
        for key, tensor in samples.items():
            rows = tensor.detach().float().reshape(-1, tensor.shape[-1])
            self.tables[key] = _kmeans_per_channel(
                rows, 2**self.config.bits, self.config.calibration_iters
            ).cpu()

    def save(self, path: str) -> None:
        torch.save(
            {
                "bits": self.config.bits,
                "tables": {f"{layer}:{kind}": t for (layer, kind), t in self.tables.items()},
            },
            path,
        )

    def load(self, path: str) -> None:
        stored = torch.load(path, map_location="cpu")
        if stored["bits"] != self.config.bits:
            raise ValueError(
                f"Tables in {path} are {stored['bits']}-bit, not {self.config.bits}-bit"
            )
        for name, table in stored["tables"].items():
            layer, kind = name.split(":")
            self.tables[(int(layer), kind)] = table.float()


def _kmeans_per_channel(rows: torch.Tensor, levels: int, iterations: int) -> torch.Tensor:
    """1-D k-means for every channel at once: ``rows`` [samples, channels] ->
    ascending centroids [channels, levels]. Starts at evenly spaced quantiles."""
    quantiles = (torch.arange(levels, dtype=torch.float32) + 0.5) / levels
    centroids = torch.quantile(rows, quantiles.to(rows.device), dim=0).T.contiguous()
    for _ in range(iterations):
        assign = nearest_level(rows, centroids)  # [samples, channels]
        one_hot = torch.nn.functional.one_hot(assign, levels).float()
        counts = one_hot.sum(dim=0)  # [channels, levels]
        sums = (one_hot * rows[..., None]).sum(dim=0)
        updated = sums / counts.clamp(min=1)
        centroids = torch.where(counts > 0, updated, centroids).sort(dim=-1).values
    return centroids
