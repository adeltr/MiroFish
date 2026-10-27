"""Règle de base du darwinisme : les engagements suivent la performance passée."""

import math
from typing import Dict, Mapping, Sequence


def trailing_score(monthly_returns: Sequence[float], window: int = 36) -> float:
    """Rendement cumulé sur les `window` derniers mois."""
    growth = 1.0
    for r in monthly_returns[-window:]:
        growth *= 1.0 + r
    return growth - 1.0


def fundraising_shares(scores: Mapping[str, float], temperature: float = 0.1) -> Dict[str, float]:
    """Parts de levée par softmax des scores. Température haute : quasi uniforme ;
    basse : le meilleur fonds capte presque tout."""
    if not scores:
        return {}
    t = max(temperature, 1e-6)
    top = max(scores.values())
    exps = {k: math.exp((v - top) / t) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}
