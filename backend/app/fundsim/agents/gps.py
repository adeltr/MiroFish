"""Gérants (GP) fictifs et univers de fonds investissables."""

import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

from ..engine import Fund, FundParams


@dataclass(frozen=True)
class GPSpec:
    gp_id: str
    name: str
    strategy: str
    theme: str
    alpha_annual: float        # compétence supposée (rendement net additionnel annuel)
    fee_annual: float          # frais de gestion annuels (déduits du rendement)
    size: float                # taille cible d'un fonds, M$
    first_close: int           # mois (0 = début du scénario) d'ouverture du premier fonds


GPS: Tuple[GPSpec, ...] = (
    GPSpec('atlas', 'Atlas Capital Partners', 'buyout', 'diversified', 0.030, 0.020, 12_000, 2),
    GPSpec('ridgeline', 'Ridgeline Equity', 'buyout', 'industrials', 0.015, 0.020, 3_000, 14),
    GPSpec('northbeam', 'Northbeam Healthcare Partners', 'buyout', 'healthcare', 0.025, 0.020, 4_000, 8),
    GPSpec('summit', 'Summit Growth', 'growth', 'tech', 0.020, 0.020, 3_500, 5),
    GPSpec('evergreen', 'Evergreen Climate Growth', 'growth', 'energy_transition', 0.010, 0.020, 2_500, 20),
    GPSpec('lumen', 'Lumen Ventures', 'venture', 'tech', 0.040, 0.025, 1_500, 0),
    GPSpec('helix', 'Helix Bio Ventures', 'venture', 'healthcare', -0.010, 0.025, 1_200, 11),
    GPSpec('meridian', 'Meridian Secondaries', 'secondaries', 'diversified', 0.025, 0.015, 6_000, 6),
    GPSpec('harbor', 'Harbor GP-Led Solutions', 'secondaries', 'diversified', 0.015, 0.015, 3_000, 26),
    GPSpec('keystone', 'Keystone Credit', 'private_credit', 'financials', 0.010, 0.010, 5_000, 3),
    GPSpec('cobalt', 'Cobalt Credit Opportunities', 'private_credit', 'diversified', 0.020, 0.012, 2_500, 17),
    GPSpec('greenline', 'Greenline Energy Transition', 'infrastructure', 'energy_transition', 0.015, 0.015, 4_500, 9),
    GPSpec('bedrock', 'Bedrock Infrastructure', 'infrastructure', 'real_assets', 0.010, 0.015, 7_000, 12),
    GPSpec('landmark', 'Landmark Real Assets', 'real_estate', 'real_assets', -0.005, 0.015, 3_000, 4),
)

NOISE_MONTHLY = {'venture': 0.010, 'growth': 0.007, 'buyout': 0.005, 'secondaries': 0.004,
                 'private_credit': 0.003, 'infrastructure': 0.003, 'real_estate': 0.005}
FUND_CYCLE_MONTHS = 36      # un nouveau fonds par GP tous les trois ans
OPEN_WINDOW_MONTHS = 12     # durée pendant laquelle un fonds accepte des engagements


def build_universe(horizon: int = 60) -> Tuple[Dict[str, Fund], Dict[str, GPSpec]]:
    """Tous les fonds qui ouvrent pendant le scénario (+ un fonds « legacy » par stratégie)."""
    funds: Dict[str, Fund] = {}
    for gp in GPS:
        n, month = 1, gp.first_close
        while month < horizon:
            fid = f"{gp.gp_id}-{n}"
            funds[fid] = Fund(fid, gp.gp_id, gp.strategy, gp.theme, month, gp.size,
                              FundParams(life_years=10.0 if gp.strategy != 'venture' else 12.0))
            n, month = n + 1, month + FUND_CYCLE_MONTHS
    return funds, {g.gp_id: g for g in GPS}


def open_funds(funds: Mapping[str, Fund], month: int) -> List[Fund]:
    """Fonds qui acceptent des engagements ce mois-ci."""
    return sorted((f for f in funds.values()
                   if f.gp != 'legacy' and f.vintage_month <= month < f.vintage_month + OPEN_WINDOW_MONTHS),
                  key=lambda f: f.fund_id)


def fund_return(fund: Fund, gp: GPSpec, strategy_return: float, month: int, seed: int) -> float:
    """Rendement net mensuel d'un fonds : stratégie + (alpha - frais)/12 + bruit déterministe."""
    rng = random.Random(f"{seed}|{fund.fund_id}|{month}")
    noise = rng.gauss(0.0, NOISE_MONTHLY[fund.strategy])
    return strategy_return + (gp.alpha_annual - gp.fee_annual) / 12.0 + noise
