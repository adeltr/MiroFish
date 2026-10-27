"""Fonds, positions et comptes de LP."""

from dataclasses import dataclass, field
from typing import Dict, Mapping

from .cashflows import FundParams, call_amount, distribution_amount

STRATEGIES = (
    'buyout', 'growth', 'venture', 'secondaries',
    'private_credit', 'infrastructure', 'real_estate',
)


@dataclass(frozen=True)
class Fund:
    fund_id: str
    gp: str
    strategy: str
    theme: str
    vintage_month: int
    target_size: float
    params: FundParams = FundParams()


@dataclass
class Position:
    fund_id: str
    commitment: float
    unfunded: float
    nav: float = 0.0
    called: float = 0.0
    distributed: float = 0.0
    age_months: int = 0

    def step(self, fund_return: float, p: FundParams) -> tuple:
        """Avance d'un mois. Retourne (appel, distribution)."""
        grown = self.nav * (1.0 + fund_return)
        call = call_amount(self.unfunded, self.age_months, p)
        dist = distribution_amount(grown, self.age_months, p)
        self.nav = grown - dist + call
        self.unfunded -= call
        self.called += call
        self.distributed += dist
        self.age_months += 1
        if self.age_months >= p.invest_years * 12:
            self.unfunded = 0.0   # l'engagement non appelé expire
        return call, dist

    @property
    def tvpi(self) -> float:
        return (self.nav + self.distributed) / self.called if self.called else 0.0


@dataclass(frozen=True)
class MonthResult:
    calls: float
    distributions: float
    private_nav: float
    liquid: float
    unmet_calls: float


@dataclass
class LPAccount:
    lp_id: str
    liquid: float
    positions: Dict[str, Position] = field(default_factory=dict)
    unmet_calls: float = 0.0     # appels qu'il n'a pas pu financer (stress de liquidité)

    def commit(self, fund_id: str, amount: float) -> None:
        if amount <= 0:
            return
        pos = self.positions.get(fund_id)
        if pos is None:
            self.positions[fund_id] = Position(fund_id, amount, amount)
        else:
            pos.commitment += amount
            pos.unfunded += amount

    @property
    def private_nav(self) -> float:
        return sum(p.nav for p in self.positions.values())

    @property
    def unfunded(self) -> float:
        return sum(p.unfunded for p in self.positions.values())

    @property
    def total_assets(self) -> float:
        return self.liquid + self.private_nav

    def exposure(self, funds: Mapping[str, Fund], key: str) -> Dict[str, float]:
        """Part de l'exposition privée (NAV + non-appelé) par `strategy` ou `theme`."""
        out: Dict[str, float] = {}
        for fid, pos in self.positions.items():
            k = getattr(funds[fid], key)
            out[k] = out.get(k, 0.0) + pos.nav + pos.unfunded
        total = sum(out.values())
        return {k: v / total for k, v in out.items()} if total else {}

    def step_month(self, funds: Mapping[str, Fund], fund_returns: Mapping[str, float],
                   public_return: float = 0.0) -> MonthResult:
        self.liquid *= 1.0 + public_return
        calls = dists = 0.0
        for fid, pos in self.positions.items():
            c, d = pos.step(fund_returns.get(fid, 0.0), funds[fid].params)
            calls += c
            dists += d
        self.liquid += dists - calls
        if self.liquid < 0:
            self.unmet_calls += -self.liquid
            self.liquid = 0.0
        return MonthResult(calls, dists, self.private_nav, self.liquid, self.unmet_calls)
