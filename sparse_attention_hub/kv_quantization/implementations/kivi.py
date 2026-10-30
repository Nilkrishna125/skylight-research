"""KIVI preset (Liu et al., ICML 2024) and its configuration."""

from dataclasses import dataclass, field
from typing import Any

from ..base import KVQuantization, KVQuantizationConfig
from ..policies.cache.base import RegionConfig
from ..policies.cache.implementations.sink_and_local import (
    SinkAndLocalCachePolicyConfig,
)
from ..quantizers.base import QuantGranularity
from ..quantizers.implementations.integer import IntQuantizerConfig
from ..registry import KVQuantizationRegistry


@dataclass(frozen=True)
class KIVIConfig(KVQuantizationConfig):
    """KIVI: tuning-free asymmetric low-bit KV cache quantization.

    Defaults are KIVI-2 from the paper:

    - keys: 2-bit asymmetric integers, **per channel** (each channel's values over
      groups of 32 tokens share a scale and minimum);
    - values: 2-bit asymmetric integers, **per token** (groups of 32 channels);
    - buffering (``SinkAndLocalCachePolicyConfig``): keys wait in full precision
      until R = 128 have collected and are then quantized together; values keep a
      rolling full-precision window of the newest R = 128;
    - scales and minimums are stored in FP16.

    Every field can be changed for experiments (other bits, group sizes, R, sinks,
    transforms, precision policies, parameter dtypes). ``KIVI`` checks only what
    makes it KIVI; ``with_settings`` builds the common variants.
    """

    key_quantizer: IntQuantizerConfig = field(
        default_factory=lambda: IntQuantizerConfig(
            bits=2, granularity=QuantGranularity.PER_CHANNEL
        )
    )
    value_quantizer: IntQuantizerConfig = field(
        default_factory=lambda: IntQuantizerConfig(bits=2)
    )
    cache_policy: SinkAndLocalCachePolicyConfig = field(
        default_factory=SinkAndLocalCachePolicyConfig
    )

    @classmethod
    def with_settings(
        cls,
        bits: int = 2,
        group_size: int = 32,
        residual_length: int = 128,
        **overrides: Any,
    ) -> "KIVIConfig":
        """KIVI with the paper's three knobs, e.g. ``with_settings(bits=4)`` for KIVI-4.

        ``overrides`` sets any other ``KVQuantizationConfig`` field (``store_codes``,
        ``error_metric``, ``key_transform``, ...).
        """
        return cls(
            key_quantizer=IntQuantizerConfig(
                bits=bits, granularity=QuantGranularity.PER_CHANNEL, group_size=group_size
            ),
            value_quantizer=IntQuantizerConfig(bits=bits, group_size=group_size),
            cache_policy=SinkAndLocalCachePolicyConfig(
                key_local=RegionConfig(residual_length),
                value_local=RegionConfig(residual_length),
            ),
            **overrides,
        )


@KVQuantizationRegistry.register(KIVIConfig)
class KIVI(KVQuantization):
    """The KIVI recipe: generic components with KIVI's settings and checks.

    Quantization, buffering and storage are the shared components (IntQuantizer,
    SinkAndLocalCachePolicy, QuantizedKVStore); this class only validates that the
    configuration is a KIVI configuration before building them.
    """

    def __init__(self, config: KIVIConfig) -> None:
        validate_kivi(config)
        super().__init__(config)


def validate_kivi(config: KVQuantizationConfig) -> None:
    """Raise if ``config`` breaks what defines KIVI (other settings are free)."""
    keys, values = config.key_quantizer, config.value_quantizer
    if not isinstance(keys, IntQuantizerConfig) or not isinstance(
        values, IntQuantizerConfig
    ):
        raise ValueError("KIVI uses integer quantizers (IntQuantizerConfig) for K and V")
    if QuantGranularity(keys.granularity) != QuantGranularity.PER_CHANNEL:
        raise ValueError("KIVI quantizes keys per channel (PER_CHANNEL)")
    if QuantGranularity(values.granularity) == QuantGranularity.PER_CHANNEL:
        raise ValueError("KIVI quantizes values per token (PER_TOKEN or PER_BLOCK)")
    if not isinstance(config.cache_policy, SinkAndLocalCachePolicyConfig):
        raise ValueError(
            "KIVI buffers keys and keeps a rolling value window; use "
            "SinkAndLocalCachePolicyConfig"
        )
    residual = config.cache_policy.key_local.tokens
    if residual % keys.group_size:
        raise ValueError(
            f"KIVI needs the key buffer R ({residual}) to be a multiple of the key "
            f"group size ({keys.group_size}), so every key group is complete"
        )
