"""Boucle mensuelle : marché -> comptes LP -> décisions de revue -> engagements."""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from ..agents import Population, allocation_status, fund_return, is_review_month, open_funds
from ..agents.gps import prior_history
from ..data.scenario import Scenario
from ..llm import FundSimLLM
from .decisions import Decision, decide, make_candidates
from .social import Feed, Post, build_graph

MIN_COMMITMENT = 1.0     # M$ : en dessous, un engagement est un résidu de capacité, pas une décision
NEWS_LOOKBACK = 2        # mois d'actualité montrés au LP en plus du mois courant


@dataclass
class MonthRecord:
    label: str
    calls: float
    distributions: float
    unmet_calls: float
    new_commitments: Dict[str, float]          # par GP
    blocked: Dict[str, str]                    # LP -> raison (budget nul)
    decisions: List[Decision] = field(default_factory=list)


@dataclass
class SimulationResult:
    seed: int
    mode: str
    data_source: str
    anonymized: bool
    months: List[MonthRecord]
    commitments_by_gp: Dict[str, float]
    commitments_by_strategy: Dict[str, float]
    commitments_by_theme: Dict[str, float]
    final_allocations: Dict[str, Dict[str, float]]    # archétype -> stratégie -> part
    target_allocations: Dict[str, Dict[str, float]]
    decision_sources: Dict[str, int]
    llm_calls: int
    social_enabled: bool = False
    feed: List[dict] = field(default_factory=list)
    influence: Dict[str, float] = field(default_factory=dict)
    follows: Dict[str, List[str]] = field(default_factory=dict)
    social_backend: str = 'off'
    oasis: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def run_simulation(pop: Population, scenario: Scenario, llm: Optional[FundSimLLM] = None,
                   data_source: str = 'unknown', model: Optional[str] = None, seed: Optional[int] = None,
                   social='native', oasis_reactions: str = 'rule', oasis_llm_budget: int = 150,
                   oasis_db_dir: Optional[str] = None) -> SimulationResult:
    """`llm=None` : tous les LP suivent la règle de référence (aucun appel réseau).
    `social` : 'native' (fil léger), 'oasis' (plateforme OASIS) ou False/'off'."""
    seed = pop.seed if seed is None else seed
    kind = 'native' if social is True else 'off' if social in (False, None, 'off') else social
    feed = _make_feed(kind, pop, seed, oasis_reactions, oasis_llm_budget, oasis_db_dir)
    try:
        return _simulate(pop, scenario, llm, data_source, model, seed, feed)
    finally:
        if hasattr(feed, 'close'):
            feed.close()


def _make_feed(kind: str, pop: Population, seed: int, reactions: str, llm_budget: int,
               db_dir: Optional[str]):
    if kind == 'off':
        return None
    graph = build_graph(pop, seed)
    if kind == 'native':
        return Feed(graph)
    if kind == 'oasis':
        from .oasis_social import OasisFeed
        return OasisFeed(graph, agents=pop.lps, seed=seed, reactions=reactions,
                         llm_action_budget=llm_budget, db_dir=db_dir)
    raise ValueError("social ∈ {native, oasis, off}")


def _simulate(pop: Population, scenario: Scenario, llm: Optional[FundSimLLM], data_source: str,
              model: Optional[str], seed: int, feed) -> SimulationResult:
    funds, gps = pop.funds, pop.gps
    history: Dict[str, List[float]] = prior_history(gps, seed)
    raised: Dict[str, float] = {}
    by_gp: Dict[str, float] = {}
    by_strategy: Dict[str, float] = {}
    by_theme: Dict[str, float] = {}
    sources: Dict[str, int] = {}
    themes = sorted({f.theme for f in funds.values()})
    records: List[MonthRecord] = []
    n = len(pop.lps)
    for month in scenario.months:
        i = month.index
        # 1. rendements du mois pour chaque fonds existant
        returns: Dict[str, float] = {}
        for fid, f in funds.items():
            strat_ret = month.strategy_returns[f.strategy]
            if f.gp == 'legacy':
                returns[fid] = strat_ret
            elif f.vintage_month <= i:
                returns[fid] = fund_return(f, gps[f.gp], strat_ret, i, seed)
                history.setdefault(fid, []).append(returns[fid])
        # 2. les comptes avancent d'un mois
        calls = dists = 0.0
        for lp in pop.lps:
            res = lp.account.step_month(funds, returns, month.public_return)
            calls += res.calls
            dists += res.distributions
        # 3. revues et engagements (ordre tournant pour ne favoriser personne)
        recent = [dict(nw, month=sm.label) for sm in scenario.months[max(0, i - NEWS_LOOKBACK): i + 1]
                  for nw in sm.news]
        rec = MonthRecord(month.label, calls, dists, sum(l.account.unmet_calls for l in pop.lps), {}, {})
        for lp in sorted(pop.lps, key=lambda a: (a.index - i) % n):
            if not is_review_month(lp.index, i, lp.policy.review_every):
                continue
            status = allocation_status(lp.account, lp.policy)
            if status.budget <= 0:
                rec.blocked[lp.agent_id] = status.reason
                continue
            candidates = make_candidates(open_funds(funds, i), gps, history, raised)
            if not candidates:
                continue
            view = feed.peer_view(lp.agent_id, i) if feed else None
            dec = decide(lp, status, candidates, month, recent, llm, themes, model,
                         peer_summary=view.summary(funds, lp.policy.herding) if view else "",
                         peer_weights=view.weights if view else None)
            sources[dec.source] = sources.get(dec.source, 0) + 1
            for fid, w in dec.weights.items():
                amount = min(w * status.budget, funds[fid].target_size - raised.get(fid, 0.0))
                if amount < MIN_COMMITMENT:
                    continue
                lp.account.commit(fid, amount)
                raised[fid] = raised.get(fid, 0.0) + amount
                f = funds[fid]
                if feed:
                    feed.publish(Post(i, month.label, lp.agent_id, fid, w, amount, dec.rationale
                                      or f"Engagement de {amount:,.0f} M$ dans {fid} ({f.strategy}, {f.theme})."))
                rec.new_commitments[f.gp] = rec.new_commitments.get(f.gp, 0.0) + amount
                by_gp[f.gp] = by_gp.get(f.gp, 0.0) + amount
                by_strategy[f.strategy] = by_strategy.get(f.strategy, 0.0) + amount
                by_theme[f.theme] = by_theme.get(f.theme, 0.0) + amount
            rec.decisions.append(dec)
        if hasattr(feed, 'end_of_month'):
            feed.end_of_month(i)
        records.append(rec)

    return SimulationResult(
        seed=seed, mode='llm' if llm is not None else 'rule', data_source=data_source,
        anonymized=scenario.anonymized, months=records, commitments_by_gp=by_gp,
        commitments_by_strategy=by_strategy, commitments_by_theme=by_theme,
        final_allocations=_final_allocations(pop), target_allocations=_targets(pop),
        decision_sources=sources, llm_calls=llm.calls if llm is not None else 0,
        social_enabled=feed is not None, feed=feed.export() if feed else [],
        influence={k: round(v, 3) for k, v in feed.graph.influence.items()} if feed else {},
        follows=feed.graph.follows if feed else {},
        social_backend=('oasis' if hasattr(feed, 'oasis_export') else 'native') if feed else 'off',
        oasis=feed.oasis_export() if hasattr(feed, 'oasis_export') else {})


def _final_allocations(pop: Population) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, Dict[str, float]] = {}
    for lp in pop.lps:
        bucket = totals.setdefault(lp.archetype, {})
        for fid, pos in lp.account.positions.items():
            s = pop.funds[fid].strategy
            bucket[s] = bucket.get(s, 0.0) + pos.nav + pos.unfunded
    out = {}
    for arch, b in totals.items():
        tot = sum(b.values())
        out[arch] = {s: round(v / tot, 4) for s, v in sorted(b.items())} if tot else {}
    return out


def _targets(pop: Population) -> Dict[str, Dict[str, float]]:
    return {lp.archetype: dict(lp.policy.strategy_targets) for lp in pop.lps}
