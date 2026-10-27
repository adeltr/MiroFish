"""Appels de fonds et distributions, d'après le modèle Takahashi-Alexander (Yale), au pas mensuel."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FundParams:
    life_years: float = 10.0        # durée de vie du fonds
    invest_years: float = 5.0       # période d'investissement : les appels s'arrêtent ensuite
    call_rate: float = 0.25         # part annuelle du non-appelé qui est appelée
    bow: float = 2.5                # convexité des distributions (plus haut = plus tard)
    yield_: float = 0.0             # taux de distribution minimal annuel


def monthly_rate(annual: float) -> float:
    """Taux mensuel équivalent à un taux annuel de « part prélevée » : (1-m)^12 = 1-annuel."""
    annual = min(max(annual, 0.0), 1.0)
    return 1.0 - (1.0 - annual) ** (1.0 / 12.0)


def call_amount(unfunded: float, age_months: int, p: FundParams) -> float:
    if age_months >= p.invest_years * 12:
        return 0.0
    return min(unfunded, unfunded * monthly_rate(p.call_rate))


def distribution_rate_annual(age_months: int, p: FundParams) -> float:
    age_years = age_months / 12.0
    if age_years >= p.life_years:
        return 1.0
    return min(1.0, max(p.yield_, (age_years / p.life_years) ** p.bow))


def distribution_amount(nav_grown: float, age_months: int, p: FundParams) -> float:
    return nav_grown * monthly_rate(distribution_rate_annual(age_months, p))
