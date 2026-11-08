"""Replenishment policy (FR-5). Pure functions, easy to test and to explain in an interview.

Notation: mu = mean daily demand, sigma = std of daily demand, L = lead time (days),
R = review period (days; 1 = reviewed every day), z = normal quantile of the service level.
"""
from __future__ import annotations

import math

from scipy.stats import norm


def z_from_service_level(service_level: float) -> float:
    if not 0.5 <= service_level <= 0.999:
        raise ValueError("service_level must be within [0.5, 0.999]")
    return float(norm.ppf(service_level))


def reorder_point(mu: float, sigma: float, lead_time: int, z: float) -> float:
    """ROP = mu*L + z*sigma*sqrt(L): expected demand during lead time + safety stock."""
    return mu * lead_time + z * sigma * math.sqrt(lead_time)


def order_up_to_level(mu: float, sigma: float, lead_time: int, z: float, review: int = 1) -> float:
    """S = mu*(L+R) + z*sigma*sqrt(L+R): the level we refill to when reviewing every R days."""
    horizon = lead_time + review
    return mu * horizon + z * sigma * math.sqrt(horizon)


def order_quantity(inventory_position: float, target: float, moq: int = 0) -> int:
    """Units to order now. 0 when stock is enough; otherwise at least MOQ, integer (BR-3)."""
    if moq < 0:
        raise ValueError("moq must be >= 0")
    gap = target - inventory_position
    if gap <= 0:
        return 0
    return int(max(moq, math.ceil(gap)))
