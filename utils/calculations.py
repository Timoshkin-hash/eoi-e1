"""Вспомогательные расчёты для оценки проекта внедрения CRM."""

from __future__ import annotations

import numpy as np


def npv(discount_rate: float, cash_flows: list[float]) -> float:
    """Возвращает чистую приведённую стоимость потока, где первый элемент — год 0."""
    periods = np.arange(len(cash_flows))
    return float(np.sum(np.asarray(cash_flows) / (1 + discount_rate) ** periods))


def simple_payback(initial_outlay: float, annual_cash_flow: float) -> float:
    """Возвращает простой срок окупаемости в годах для постоянного годового потока."""
    if annual_cash_flow <= 0:
        return float("inf")
    return initial_outlay / annual_cash_flow
