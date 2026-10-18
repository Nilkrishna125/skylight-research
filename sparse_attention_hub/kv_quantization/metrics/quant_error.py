"""Optional reconstruction diagnostics on compressed regions."""

import torch

from sparse_attention_hub.metric_logging.logger import MicroMetricLogger

from ..cache.state import TensorCacheState

METRIC_IDS = (
    "kv_quantization_key_error",
    "kv_quantization_value_error",
    "kv_quantization_store_bytes",
)

for identifier in METRIC_IDS:
    MicroMetricLogger.register_metric(
        identifier, int if identifier.endswith("bytes") else float
    )


class QuantErrorMetric:
    @staticmethod
    def compute(original: torch.Tensor, dequantized: torch.Tensor) -> float:
        reference = original.float()
        return float(
            (
                torch.linalg.vector_norm(reference - dequantized.float())
                / torch.linalg.vector_norm(reference).clamp_min(1e-12)
            ).item()
        )

    def log(
        self,
        kind: str,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        state: TensorCacheState,
        layer_idx: int,
    ) -> None:
        # Exclude FP sink and recent regions; evaluate only encoded token intervals.
        if not state.chunks:
            return
        starts = [(c.start_token, c.start_token + c.token_count) for c in state.chunks]
        before = torch.cat([original[..., a:b, :] for a, b in starts], dim=-2)
        after = torch.cat([reconstructed[..., a:b, :] for a, b in starts], dim=-2)
        MicroMetricLogger().log(
            f"kv_quantization_{kind}_error",
            self.compute(before, after),
            metadata={"layer_idx": layer_idx, "tokens": before.shape[-2]},
        )
