"""Kiểm thử công thức Precision và Recall cơ bản."""

from src.metrics import precision_recall


def test_precision_recall():
    precision, recall = precision_recall(tp=8, fp=2, fn=4)
    assert precision == 0.8
    assert recall == 8 / 12
