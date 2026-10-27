"""Construit une population reproductible : LP (avec portefeuille initial), GP et fonds."""

import random
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from ..engine import Fund, LPAccount, Position
from .archetypes import ARCHETYPES, DEFAULT_COUNTS
from .gps import GPSpec, build_universe
from .policy import LPPolicy

LEGACY_AGE_MONTHS = 48
SUPPLY_RATIO = 1.5         # capacité totale des fonds = 1,5 x la demande des LP, par stratégie : de la concurrence
INITIAL_FILL = 0.8         # le LP démarre à 80 % de sa cible privée : il a de la marge pour s'engager


@dataclass
class LPAgent:
    agent_id: str
    index: int
    name: str
    archetype: str
    region: str
    policy: LPPolicy
    account: LPAccount = field(repr=False)


@dataclass
class Population:
    seed: int
    lps: List[LPAgent]
    gps: Dict[str, GPSpec]
    funds: Dict[str, Fund]

    def summary(self) -> dict:
        return {
            'seed': self.seed,
            'lps': [{'id': a.agent_id, 'name': a.name, 'archetype': a.archetype, 'region': a.region,
                     'aum': round(a.account.total_assets), 'target_private': a.policy.target_private}
                    for a in self.lps],
            'gps': [{'id': g.gp_id, 'name': g.name, 'strategy': g.strategy, 'theme': g.theme}
                    for g in self.gps.values()],
            'funds': len(self.funds),
        }


def _legacy_funds() -> Dict[str, Fund]:
    from .archetypes import S
    return {f"LEG-{s}": Fund(f"LEG-{s}", 'legacy', s, 'diversified', -LEGACY_AGE_MONTHS, 0.0) for s in S}


def build_population(seed: int = 7, counts: Optional[Dict[str, int]] = None,
                     horizon: int = 60) -> Population:
    rng = random.Random(seed)
    funds, gps = build_universe(horizon)
    legacy = _legacy_funds()
    funds.update(legacy)
    lps: List[LPAgent] = []
    for arch, n in (counts or DEFAULT_COUNTS).items():
        spec = ARCHETYPES[arch]
        for k in range(n):
            aum = spec['aum'] * rng.uniform(0.6, 1.6)
            policy: LPPolicy = spec['policy']
            private = aum * policy.target_private * INITIAL_FILL
            account = LPAccount(f"LP{len(lps) + 1:02d}", liquid=aum - private)
            for s, w in policy.strategy_targets.items():
                nav = private * w
                if nav > 0:
                    account.positions[f"LEG-{s}"] = Position(
                        f"LEG-{s}", commitment=nav * 1.3, unfunded=nav * 0.1, nav=nav,
                        called=nav * 1.1, distributed=0.0, age_months=LEGACY_AGE_MONTHS)
            suffix = chr(ord('A') + k)
            lps.append(LPAgent(
                agent_id=account.lp_id, index=len(lps), name=f"{spec['label']} {suffix}",
                archetype=arch, region=spec['regions'][k % len(spec['regions'])],
                policy=policy, account=account))
    funds = _calibrate_supply(funds, lps, horizon)
    return Population(seed, lps, gps, funds)


def _calibrate_supply(funds: Dict[str, Fund], lps: List[LPAgent], horizon: int) -> Dict[str, Fund]:
    """Ajuste la taille des fonds pour que l'offre soit SUPPLY_RATIO fois la demande des LP, stratégie
    par stratégie. Sans cela, tout fonds se remplit quoi qu'il arrive et la sélection entre GP disparaît."""
    years = horizon / 12.0
    demand: Dict[str, float] = {}
    for lp in lps:
        # rythme annuel : le plus grand du rythme fixé et du renouvellement d'un quart de la cible
        annual = max(lp.policy.pacing_annual, 0.25 * lp.policy.target_private)
        base = annual * lp.account.total_assets * years
        for s, w in lp.policy.strategy_targets.items():
            demand[s] = demand.get(s, 0.0) + base * w
    supply: Dict[str, float] = {}
    for f in funds.values():
        if f.gp != 'legacy':
            supply[f.strategy] = supply.get(f.strategy, 0.0) + f.target_size
    scale = {s: SUPPLY_RATIO * demand.get(s, 0.0) / v for s, v in supply.items() if v > 0}
    return {fid: f if f.gp == 'legacy' else replace(f, target_size=f.target_size * scale.get(f.strategy, 1.0))
            for fid, f in funds.items()}
