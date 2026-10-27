"""Construit une population reproductible : LP (avec portefeuille initial), GP et fonds."""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..engine import Fund, LPAccount, Position
from .archetypes import ARCHETYPES, DEFAULT_COUNTS
from .gps import GPSpec, build_universe
from .policy import LPPolicy

LEGACY_AGE_MONTHS = 48
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
    return Population(seed, lps, gps, funds)
