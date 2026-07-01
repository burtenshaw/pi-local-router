from local_router.confidence import choose_route, summarize_confidence


def test_easy_metrics_route_local() -> None:
    metrics = summarize_confidence([1.0, 1.2, 1.1], [0.6, 0.55, 0.58], vocab_size=128_000)
    route, reason = choose_route(metrics, entropy_threshold=4.0, top1_threshold=0.08, confidence_threshold=0.55)

    assert route == "local"
    assert metrics.p90_entropy > metrics.avg_entropy
    assert metrics.p10_top1_prob < metrics.mean_top1_prob
    assert "confidence" in reason


def test_high_entropy_routes_remote() -> None:
    metrics = summarize_confidence([8.0, 8.2, 7.9], [0.02, 0.03, 0.02], vocab_size=128_000)
    route, reason = choose_route(metrics, entropy_threshold=4.0, top1_threshold=0.08, confidence_threshold=0.55)

    assert route == "remote"
    assert "low" in reason


def test_confidence_alone_does_not_route_local() -> None:
    metrics = summarize_confidence([0.2, 0.2], [0.93, 0.93], vocab_size=128_000)
    route, reason = choose_route(metrics, entropy_threshold=0.12, top1_threshold=0.95, confidence_threshold=0.90)

    assert metrics.confidence >= 0.90
    assert route == "remote"
    assert "low" in reason


def test_nonfinite_entropy_is_sanitized() -> None:
    metrics = summarize_confidence([float("nan")], [float("nan")], vocab_size=128_000)
    route, _reason = choose_route(metrics, entropy_threshold=4.0, top1_threshold=0.08, confidence_threshold=0.55)

    assert metrics.avg_entropy > 0
    assert metrics.mean_top1_prob == 0.0
    assert metrics.confidence == 0.0
    assert route == "remote"
