"""Validation déterministe d'une allocation proposée (par un LLM ou une règle)."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from .portfolio import Fund

EPS = 1e-9


@dataclass(frozen=True)
class Limits:
    max_by_strategy: Mapping[str, float] = field(default_factory=dict)
    max_by_theme: Mapping[str, float] = field(default_factory=dict)
    max_single_fund: float = 0.25


@dataclass
class ValidationResult:
    weights: Dict[str, float]
    unallocated: float
    adjustments: List[str]


def _clean(proposal: Mapping[str, float], funds: Mapping[str, Fund],
           adjustments: List[str]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for fid, w in proposal.items():
        if fid not in funds:
            adjustments.append(f"{fid}: fonds inconnu, ignoré")
        elif not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0:
            adjustments.append(f"{fid}: poids invalide ({w!r}), ignoré")
        else:
            cleaned[fid] = float(w)
    total = sum(cleaned.values())
    return {k: v / total for k, v in cleaned.items()} if total else {}


def validate_allocation(proposal: Mapping[str, float], funds: Mapping[str, Fund], limits: Limits,
                        fund_caps: Optional[Mapping[str, float]] = None) -> ValidationResult:
    """Nettoie, normalise puis applique les plafonds (fonds, stratégie, thème).

    Les poids retirés par un plafond sont redistribués aux autres fonds selon les poids
    proposés. Si les plafonds rendent la somme de 1 impossible, le reste est « non alloué ».
    `fund_caps` permet un plafond propre à chaque fonds (ex. capacité de levée restante).
    """
    adjustments: List[str] = []
    target = _clean(proposal, funds, adjustments)
    alloc = {fid: 0.0 for fid in target}
    strat_used: Dict[str, float] = {}
    theme_used: Dict[str, float] = {}
    binding: Dict[str, str] = {}

    def headroom(fid: str) -> tuple:
        f = funds[fid]
        cap = limits.max_single_fund
        reason = 'plafond par fonds'
        if fund_caps and fid in fund_caps and fund_caps[fid] < cap:
            cap, reason = fund_caps[fid], 'capacité du fonds'
        room = cap - alloc[fid]
        s_cap = limits.max_by_strategy.get(f.strategy)
        if s_cap is not None and s_cap - strat_used.get(f.strategy, 0.0) < room:
            room, reason = s_cap - strat_used.get(f.strategy, 0.0), f"plafond stratégie {f.strategy}"
        t_cap = limits.max_by_theme.get(f.theme)
        if t_cap is not None and t_cap - theme_used.get(f.theme, 0.0) < room:
            room, reason = t_cap - theme_used.get(f.theme, 0.0), f"plafond thème {f.theme}"
        return max(room, 0.0), reason

    remaining = 1.0 if target else 0.0
    for _ in range(50):
        if remaining < EPS:
            break
        open_ids = [fid for fid in target if headroom(fid)[0] > EPS]
        base_total = sum(target[fid] for fid in open_ids)
        if not open_ids or base_total <= 0:
            break
        given_round = 0.0
        for fid in sorted(open_ids, key=lambda k: (-target[k], k)):
            want = remaining * target[fid] / base_total
            room, reason = headroom(fid)
            give = min(want, room)
            if give < want - EPS:
                binding[fid] = reason
            alloc[fid] += give
            strat_used[funds[fid].strategy] = strat_used.get(funds[fid].strategy, 0.0) + give
            theme_used[funds[fid].theme] = theme_used.get(funds[fid].theme, 0.0) + give
            given_round += give
        remaining -= given_round
        if given_round < EPS:
            break

    for fid, why in binding.items():
        adjustments.append(f"{fid}: {target[fid]:.1%} proposé, {alloc[fid]:.1%} retenu ({why})")
    weights = {fid: w for fid, w in alloc.items() if w > EPS}
    return ValidationResult(weights, max(remaining, 0.0) if target else 1.0, adjustments)
