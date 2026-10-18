"""Encoded tensor records and individual stored chunks."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from ..context import AttentionContext


@dataclass
class QuantizedTensor:
    codes: torch.Tensor
    scales: torch.Tensor
    zero_points: Optional[torch.Tensor]
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    axis: int
    bits: int
    format_id: str
    group_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    norms: Optional[torch.Tensor] = None
    global_scale: Optional[torch.Tensor] = None
    codebook: Optional[torch.Tensor] = None
    outlier_indices: Optional[torch.Tensor] = None
    outlier_values: Optional[torch.Tensor] = None

    def storage_bytes(self) -> int:
        """Tensor storage only; excludes Python objects and shared calibration data."""
        tensors = [
            self.codes,
            self.scales,
            self.zero_points,
            self.norms,
            self.global_scale,
            self.codebook,
            self.outlier_indices,
            self.outlier_values,
        ]
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)


@dataclass
class StoredChunk:
    start_token: int
    token_count: int
    context: AttentionContext
    transform_state: Any = None
    quantized: Optional[QuantizedTensor] = None
    dequantized: Optional[torch.Tensor] = None
