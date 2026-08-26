"""Kiểm thử hàm tổng hợp latency, không cần GPU hoặc checkpoint."""

import pytest

from tools.benchmark_inference import summarize_latencies


def test_summarize_latencies_reports_ms_and_fps():
    result = summarize_latencies([0.1, 0.2, 0.3])

    assert result["mean_ms"] == pytest.approx(200.0)
    assert result["median_ms"] == pytest.approx(200.0)
    assert result["p95_ms"] == pytest.approx(290.0)
    assert result["fps_from_mean_latency"] == pytest.approx(5.0)


def test_summarize_latencies_rejects_empty_or_invalid_values():
    with pytest.raises(ValueError):
        summarize_latencies([])
    with pytest.raises(ValueError):
        summarize_latencies([0.0])
