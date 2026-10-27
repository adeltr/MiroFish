"""Politique d'investissement d'un LP et règles déterministes qui encadrent ses décisions."""

from dataclasses import dataclass, field
from typing import Mapping, Tuple

from ..engine import LPAccount, Limits


GAP_CLOSE = 0.25   # part de l'écart à la cible comblée à chaque revue


@dataclass(frozen=True)
class LPPolicy:
    target_private: float                  # part cible du privé (NAV + non-appelé) dans l'actif total
    band: Tuple[float, float]              # bornes de la politique (plancher, plafond)
    strategy_targets: Mapping[str, float]  # répartition cible du privé par stratégie (somme = 1)
    pacing_annual: float                   # nouveaux engagements annuels, en part de l'actif total
    liquidity_floor: float                 # part minimale d'actifs liquides
    max_single_fund: float                 # part maximale d'un budget d'engagement dans un fonds
    max_theme: float = 0.35                # part maximale d'un thème
    theme_tilts: Mapping[str, float] = field(default_factory=dict)   # thèmes préférés (>1) ou évités (<1)
    herding: float = 0.4                   # 0 = indépendant, 1 = suit les autres LP
    review_every: int = 3                  # mois entre deux revues d'allocation
    persona: str = ""

    def limits(self) -> Limits:
        caps = {s: min(1.0, t * 1.6 + 0.05) for s, t in self.strategy_targets.items()}
        return Limits(max_by_strategy=caps, max_by_theme={}, max_single_fund=self.max_single_fund)

    def theme_limits(self, themes) -> Limits:
        base = self.limits()
        return Limits(base.max_by_strategy, {t: self.max_theme for t in themes}, base.max_single_fund)


@dataclass(frozen=True)
class Status:
    exposure_share: float
    liquid_share: float
    state: str            # 'under' | 'in_band' | 'over'
    budget: float         # montant que le LP peut engager à cette revue
    reason: str


def is_review_month(agent_index: int, month_index: int, review_every: int) -> bool:
    """Revues échelonnées : les LP ne décident pas tous le même mois (et limitent les appels LLM)."""
    return (month_index - agent_index) % review_every == 0


def allocation_status(account: LPAccount, policy: LPPolicy) -> Status:
    total = account.total_assets
    if total <= 0:
        return Status(0.0, 0.0, 'under', 0.0, 'actif nul')
    exposure = (account.private_nav + account.unfunded) / total
    liquid = account.liquid / total
    lo, hi = policy.band
    state = 'over' if exposure >= hi else 'under' if exposure < lo else 'in_band'
    # Liquidité mesurée après paiement de tous les appels de fonds à venir.
    if (account.liquid - account.unfunded) / total < policy.liquidity_floor:
        return Status(exposure, liquid, state, 0.0, "liquidités (après appels futurs) sous le plancher")
    if state == 'over':
        return Status(exposure, liquid, state, 0.0, "au-dessus du plafond de la politique (effet dénominateur)")
    pace = policy.pacing_annual * total * policy.review_every / 12.0
    held = account.private_nav + account.unfunded
    gap = max(policy.target_private * total - held, 0.0)   # sur-engagement : on comble une partie de l'écart
    room = hi * total - held
    return Status(exposure, liquid, state, max(min(pace + GAP_CLOSE * gap, room), 0.0), "ok")


def prompt_block(name: str, archetype: str, policy: LPPolicy, status: Status) -> str:
    """Description de mandat en français, destinée au prompt du LLM (étape 4)."""
    targets = ", ".join(f"{s} {w:.0%}" for s, w in policy.strategy_targets.items() if w > 0)
    tilts = ", ".join(f"{t} (x{m:g})" for t, m in policy.theme_tilts.items()) or "aucun"
    return (
        f"Tu es {name}, un investisseur institutionnel de type « {archetype} ».\n{policy.persona}\n"
        f"Politique : privé cible {policy.target_private:.0%} de l'actif "
        f"(bande {policy.band[0]:.0%}-{policy.band[1]:.0%}) ; répartition cible : {targets}.\n"
        f"Thèmes privilégiés ou évités : {tilts}. Liquidités minimales : {policy.liquidity_floor:.0%}. "
        f"Maximum par fonds : {policy.max_single_fund:.0%} du budget.\n"
        f"Situation : exposition privée {status.exposure_share:.1%}, liquidités {status.liquid_share:.1%}, "
        f"état {status.state}. Budget d'engagement de cette revue : {status.budget:,.0f} M$ ({status.reason})."
    )
