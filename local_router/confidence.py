from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ConfidenceMetrics:
    avg_entropy: float
    p90_entropy: float
    max_token_entropy: float
    normalized_entropy: float
    mean_top1_prob: float
    p10_top1_prob: float
    min_top1_prob: float
    confidence: float
    generated_tokens: int


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(q, 0.0), 1.0) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_confidence(entropies: Iterable[float], top1_probs: Iterable[float], vocab_size: int) -> ConfidenceMetrics:
    vocab_max_entropy = math.log(max(vocab_size, 2))
    entropy_values = [
        value if math.isfinite(value) and value >= 0.0 else vocab_max_entropy
        for value in entropies
    ]
    top1_values = [
        min(max(value, 0.0), 1.0) if math.isfinite(value) else 0.0
        for value in top1_probs
    ]
    generated_tokens = len(entropy_values)
    if generated_tokens == 0:
        return ConfidenceMetrics(
            avg_entropy=vocab_max_entropy,
            p90_entropy=vocab_max_entropy,
            max_token_entropy=vocab_max_entropy,
            normalized_entropy=1.0,
            mean_top1_prob=0.0,
            p10_top1_prob=0.0,
            min_top1_prob=0.0,
            confidence=0.0,
            generated_tokens=0,
        )

    avg_entropy = sum(entropy_values) / generated_tokens
    mean_top1 = sum(top1_values) / len(top1_values) if top1_values else 0.0
    normalized_entropy = min(max(avg_entropy / vocab_max_entropy, 0.0), 1.0)
    entropy_confidence = 1.0 - normalized_entropy
    confidence = min(max((entropy_confidence + mean_top1) / 2.0, 0.0), 1.0)
    return ConfidenceMetrics(
        avg_entropy=avg_entropy,
        p90_entropy=_quantile(entropy_values, 0.90),
        max_token_entropy=max(entropy_values),
        normalized_entropy=normalized_entropy,
        mean_top1_prob=mean_top1,
        p10_top1_prob=_quantile(top1_values, 0.10) if top1_values else 0.0,
        min_top1_prob=min(top1_values) if top1_values else 0.0,
        confidence=confidence,
        generated_tokens=generated_tokens,
    )


def choose_route(
    metrics: ConfidenceMetrics,
    entropy_threshold: float,
    top1_threshold: float,
    confidence_threshold: float,
) -> tuple[str, str]:
    entropy_ok = metrics.avg_entropy <= entropy_threshold
    top1_ok = metrics.mean_top1_prob >= top1_threshold
    confidence_ok = metrics.confidence >= confidence_threshold
    if entropy_ok and top1_ok and confidence_ok:
        return (
            "local",
            "local confidence ok: "
            f"entropy={metrics.avg_entropy:.3f}, p90_entropy={metrics.p90_entropy:.3f}, "
            f"top1={metrics.mean_top1_prob:.3f}, p10_top1={metrics.p10_top1_prob:.3f}, "
            f"confidence={metrics.confidence:.3f}",
        )
    return (
        "remote",
        "local confidence low: "
        f"entropy={metrics.avg_entropy:.3f}, p90_entropy={metrics.p90_entropy:.3f}, "
        f"top1={metrics.mean_top1_prob:.3f}, p10_top1={metrics.p10_top1_prob:.3f}, "
        f"confidence={metrics.confidence:.3f}",
    )
