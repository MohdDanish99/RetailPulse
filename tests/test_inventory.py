import math

from src.inventory.optimizer import eoq, reorder_point, safety_stock


def test_eoq_formula():
    assert math.isclose(eoq(1000, 50, 2), math.sqrt(2 * 1000 * 50 / 2))


def test_safety_stock_zero_at_50_percent():
    assert abs(safety_stock(10, 5, service_level=0.5)) < 1e-9


def test_higher_service_level_needs_more_stock():
    assert safety_stock(10, 5, 0.99) > safety_stock(10, 5, 0.90)


def test_reorder_point():
    assert reorder_point(10, 5, 20) == 70
