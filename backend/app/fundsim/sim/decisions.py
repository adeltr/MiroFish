"""Décision d'un LP : proposition (LLM ou règle de référence) puis validation déterministe."""

import dataclasses
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from ..agents import LPAgent, Status, prompt_block
from ..agents.gps import GPSpec
from ..data.scenario import MonthData
from ..engine import Fund, fundraising_shares, trailing_score, validate_allocation
from ..llm import BudgetExceeded, FundSimLLM, LLMJSONError

RECORD_WINDOW = 36
PERCEPTION_NOISE = 0.06   # écart-type du jugement propre de chaque LP sur la qualité d'un GP
SLEEVE_CAP = 0.4          # un fonds ne dépasse pas 40 % de la poche stratégique visée par le LP
HERDING_GAIN = 2.0     # un fonds plébiscité par tous les pairs voit son poids multiplié par 1 + 2 x herding


@dataclass
class Candidate:
    fund: Fund
    gp_name: str
    capacity: float                 # montant encore souscriptible, M$
    track_record: Optional[float]   # rendement cumulé du fonds précédent du GP (None = premier fonds)
    track_months: int = 0

    def view(self) -> dict:
        return {
            'fund_id': self.fund.fund_id, 'gp': self.gp_name, 'strategy': self.fund.strategy,
            'theme': self.fund.theme, 'target_size_musd': round(self.fund.target_size),
            'capacity_left_musd': round(self.capacity),
            'previous_fund_cumulative_return': None if self.track_record is None else round(self.track_record, 3),
            'previous_fund_months': self.track_months,
        }


@dataclass
class Decision:
    lp_id: str
    month: str
    source: str                     # 'llm' | 'rule' | 'fallback:<raison>'
    budget: float
    proposal: Dict[str, float]
    weights: Dict[str, float]
    unallocated: float
    adjustments: List[str] = field(default_factory=list)
    rationale: str = ""


def fund_cap(agent: LPAgent, strategy: str) -> float:
    """Part maximale du budget dans un seul fonds : le plus grand du plafond absolu de la politique
    et de SLEEVE_CAP x la cible de la stratégie (45 % de buyout visés autorisent 18 % dans un fonds)."""
    return max(agent.policy.max_single_fund, SLEEVE_CAP * agent.policy.strategy_targets.get(strategy, 0.0))


def make_candidates(open_funds_: List[Fund], gps: Mapping[str, GPSpec], history: Mapping[str, List[float]],
                    raised: Mapping[str, float]) -> List[Candidate]:
    out = []
    for f in open_funds_:
        capacity = f.target_size - raised.get(f.fund_id, 0.0)
        if capacity <= 1e-6:
            continue
        n = int(f.fund_id.rsplit('-', 1)[1])
        prev = history.get(f"{f.gp}-{n - 1}")   # le fonds n°0 est l'historique antérieur
        record = trailing_score(prev, RECORD_WINDOW) if prev else None
        out.append(Candidate(f, gps[f.gp].name, capacity, record, min(len(prev or []), RECORD_WINDOW)))
    return out


def _perceived(agent: LPAgent, c: Candidate, noise_key: str) -> float:
    base = c.track_record or 0.0
    if not noise_key:
        return base
    return base + random.Random(f"{agent.agent_id}|{c.fund.fund_id}|{noise_key}").gauss(0.0, PERCEPTION_NOISE)


def rule_proposal(agent: LPAgent, candidates: List[Candidate], temperature: float = 0.15,
                  peer_weights: Optional[Mapping[str, float]] = None, noise_key: str = "") -> Dict[str, float]:
    """Référence « darwinienne » : cible stratégique x préférences thématiques, répartie dans chaque
    stratégie selon la performance passée du GP (softmax). Avec `peer_weights`, imitation des pairs
    proportionnelle au paramètre `herding` du LP. `noise_key` non vide : chaque LP perçoit la
    performance des GP avec son propre bruit reproductible (jugement individuel)."""
    policy = agent.policy
    by_strategy: Dict[str, List[Candidate]] = {}
    for c in candidates:
        by_strategy.setdefault(c.fund.strategy, []).append(c)
    raw: Dict[str, float] = {}
    top_peer = max(peer_weights.values()) if peer_weights else 0.0
    for strat, items in by_strategy.items():
        target = policy.strategy_targets.get(strat, 0.0)
        if target <= 0:
            continue
        shares = fundraising_shares({c.fund.fund_id: _perceived(agent, c, noise_key) for c in items}, temperature)
        for c in items:
            raw[c.fund.fund_id] = target * shares[c.fund.fund_id] * policy.theme_tilts.get(c.fund.theme, 1.0)
            if top_peer > 0:
                crowd = peer_weights.get(c.fund.fund_id, 0.0) / top_peer
                raw[c.fund.fund_id] *= 1.0 + HERDING_GAIN * policy.herding * crowd
    return raw


def build_messages(agent: LPAgent, status: Status, candidates: List[Candidate], month: MonthData,
                   recent_news: List[dict], peer_summary: str = "") -> List[Dict[str, str]]:
    news = "\n".join(f"- [{n['month']}] {n['headline']} : {n['body']}" for n in recent_news) or "- (rien de notable)"
    system = ("Tu simules un investisseur institutionnel nord-américain qui répartit un budget d'engagements "
              "entre des fonds privés. Réponds uniquement par un objet JSON.")
    user = (
        f"{prompt_block(agent.name, agent.archetype, agent.policy, status)}\n\n"
        f"Date de la revue : {month.label}.\nActualité récente :\n{news}\n"
        + (f"\nCe que font les autres investisseurs :\n{peer_summary}\n" if peer_summary else "")
        + f"\nFonds ouverts aux engagements :\n{json.dumps([c.view() for c in candidates], ensure_ascii=False)}\n\n"
        'Réponds : {"weights": {"<fund_id>": <poids entre 0 et 1>, ...}, "rationale": "<40 mots maximum>"}. '
        "Les poids peuvent ne pas sommer à 1 (le reste reste en liquidités). N'utilise que les fund_id listés."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def decide(agent: LPAgent, status: Status, candidates: List[Candidate], month: MonthData,
           recent_news: List[dict], llm: Optional[FundSimLLM], themes, model: Optional[str] = None,
           peer_summary: str = "", peer_weights: Optional[Mapping[str, float]] = None) -> Decision:
    caps = {c.fund.fund_id: min(c.capacity / status.budget, fund_cap(agent, c.fund.strategy)) for c in candidates}
    source, rationale = 'rule', ''
    if llm is None:
        proposal = rule_proposal(agent, candidates, peer_weights=peer_weights, noise_key=str(month.index))
    else:
        try:
            reply = llm.complete_json(
                build_messages(agent, status, candidates, month, recent_news, peer_summary),
                model=model, temperature=0.4, max_tokens=700)
            weights = reply.get('weights')
            if not isinstance(weights, dict) or not weights:
                raise LLMJSONError("clé « weights » absente ou vide")
            proposal, source, rationale = weights, 'llm', str(reply.get('rationale', ''))[:300]
        except BudgetExceeded:
            proposal, source = rule_proposal(agent, candidates, peer_weights=peer_weights, noise_key=str(month.index)), 'fallback:budget'
        except LLMJSONError as exc:
            proposal, source = rule_proposal(agent, candidates, peer_weights=peer_weights, noise_key=str(month.index)), 'fallback:json'
            rationale = str(exc)[:120]
    limits = dataclasses.replace(agent.policy.theme_limits(themes), max_single_fund=1.0)   # les plafonds par fonds sont dans `caps`
    result = validate_allocation(proposal, {c.fund.fund_id: c.fund for c in candidates}, limits, fund_caps=caps)
    return Decision(agent.agent_id, month.label, source, status.budget,
                    {k: v for k, v in proposal.items() if isinstance(v, (int, float))},
                    result.weights, result.unallocated, result.adjustments, rationale)
