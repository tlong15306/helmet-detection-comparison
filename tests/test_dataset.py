"""Kiểm thử dataset sẽ được mở rộng sau khi có fixture COCO đại diện."""

import pytest


@pytest.mark.skip(reason="Chưa có dataset/fixture COCO đã xác nhận")
def test_dataset_placeholder():
    pass
