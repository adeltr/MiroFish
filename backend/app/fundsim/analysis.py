"""Lecture des résultats en termes d'ALLOCATIONS : par type d'investisseur, stratégie et thème,
dans le temps ; comparaison de scénarios ; comparaison à une allocation de référence.

La distance utilisée est la distance de variation totale (0 = allocations identiques, 1 = disjointes) :
la part du portefeuille qu'il faudrait déplacer pour passer de l'une à l'autre.
"""

import copy
from dataclasses import replace
from typing import Dict, List, Mapping, Optional, Sequence

from .sim.runner import SimulationResult

Weights = Mapping[str, float]


def tv_distance(a: Weights, b: Weights) -> float:
    keys = set(a) | set(b)
    return round(0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys), 4)


def hhi(amounts: Weights) -> float:
    total = sum(amounts.values())
    return round(sum((v / total) ** 2 for v in amounts.values()), 4) if total else 0.0


def trajectory(result: SimulationResult, archetype: str, dim: str = 'strategy') -> List[dict]:
    """Allocation mensuelle d'un type d'investisseur (dim = strategy | theme)."""
    attr = 'allocation_strategy' if dim == 'strategy' else 'allocation_theme'
    return [{'month': m.label, 'weights': getattr(m, attr).get(archetype, {}),
             'private_share': m.private_share.get(archetype, 0.0)} for m in result.months]


def flows_by_year(result: SimulationResult) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Nouveaux engagements (M$) par an, par stratégie et par thème. Année = préfixe du mois,
    ou « Y1.. » en mode anonymisé."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for i, m in enumerate(result.months):
        year = f"Y{i // 12 + 1}" if result.anonymized else m.label[:4]
        slot = out.setdefault(year, {'strategy': {}, 'theme': {}})
        for k, v in m.new_by_strategy.items():
            slot['strategy'][k] = round(slot['strategy'].get(k, 0.0) + v, 1)
        for k, v in m.new_by_theme.items():
            slot['theme'][k] = round(slot['theme'].get(k, 0.0) + v, 1)
    return out


def allocation_report(result: SimulationResult) -> dict:
    """Le livrable principal : où va l'argent de chaque type d'investisseur."""
    archetypes = {}
    for arch, final in result.final_allocations.items():
        target = result.target_allocations.get(arch, {})
        archetypes[arch] = {
            'private_share': {'final': result.months[-1].private_share.get(arch, 0.0),
                              'target': result.target_private_share.get(arch)},
            'strategy': {'final': final, 'target': {k: v for k, v in target.items() if v},
                         'gap_to_target': tv_distance(final, target)},
            'theme': {'final': result.final_theme_allocations.get(arch, {})},
        }
    return {
        'archetypes': archetypes,
        'flows': {'by_strategy': result.commitments_by_strategy, 'by_theme': result.commitments_by_theme,
                  'by_gp': result.commitments_by_gp, 'by_year': flows_by_year(result),
                  'gp_concentration_hhi': hhi(result.commitments_by_gp)},
        'meta': {'seed': result.seed, 'mode': result.mode, 'data_source': result.data_source,
                 'anonymized': result.anonymized, 'social_backend': result.social_backend,
                 'months': len(result.months), 'decisions': sum(result.decision_sources.values()),
                 'decision_sources': result.decision_sources, 'llm_calls': result.llm_calls},
    }


def compare(base: SimulationResult, variant: SimulationResult) -> dict:
    """Écart d'allocation entre deux runs, par type d'investisseur."""
    rows = {}
    for arch in base.final_allocations:
        mid = [tv_distance(a.allocation_strategy.get(arch, {}), b.allocation_strategy.get(arch, {}))
               for a, b in zip(base.months, variant.months)]
        rows[arch] = {
            'strategy_shift': tv_distance(base.final_allocations[arch], variant.final_allocations.get(arch, {})),
            'theme_shift': tv_distance(base.final_theme_allocations.get(arch, {}),
                                       variant.final_theme_allocations.get(arch, {})),
            'private_share_delta': round(variant.months[-1].private_share.get(arch, 0.0)
                                         - base.months[-1].private_share.get(arch, 0.0), 4),
            'mean_strategy_shift_over_time': round(sum(mid) / len(mid), 4) if mid else 0.0,
        }
    return {
        'by_archetype': rows,
        'overall': {
            'mean_strategy_shift': round(sum(r['strategy_shift'] for r in rows.values()) / len(rows), 4),
            'mean_theme_shift': round(sum(r['theme_shift'] for r in rows.values()) / len(rows), 4),
            'gp_shift': tv_distance(_share(base.commitments_by_gp), _share(variant.commitments_by_gp)),
            'gp_concentration_delta': round(hhi(variant.commitments_by_gp) - hhi(base.commitments_by_gp), 4),
        },
    }


def _share(amounts: Weights) -> Dict[str, float]:
    total = sum(amounts.values())
    return {k: v / total for k, v in amounts.items()} if total else {}


def compare_to_reference(result: SimulationResult, reference: Mapping[str, Mapping]) -> dict:
    """Écart à une allocation de référence fournie par l'utilisateur :
    {archétype: {"strategy": {stratégie: poids}, "theme": {...}, "private_share": x}}."""
    out, missing = {}, []
    for arch, ref in reference.items():
        if arch not in result.final_allocations:
            missing.append(arch)
            continue
        row = {}
        if 'strategy' in ref:
            row['strategy_gap'] = tv_distance(result.final_allocations[arch], ref['strategy'])
        if 'theme' in ref:
            row['theme_gap'] = tv_distance(result.final_theme_allocations.get(arch, {}), ref['theme'])
        if 'private_share' in ref:
            row['private_share_gap'] = round(result.months[-1].private_share.get(arch, 0.0) - ref['private_share'], 4)
        out[arch] = row
    return {'by_archetype': out, 'unknown_archetypes': missing}


def without_news(scenario):
    """Copie du scénario sans actualité (pour mesurer l'effet des news sur les allocations)."""
    clone = copy.copy(scenario)
    clone.months = [replace(m, news=[]) for m in scenario.months]
    return clone


NEWS_NOTE = ("En mode « rule », l'actualité n'influence pas les décisions : seules les variantes en mode « llm » "
             "mesurent l'effet des news sur les allocations.")


def run_variants(seed: int, series, start: str, end: str, variants: Sequence[dict], data_source: str,
                 llm_factory=None) -> Dict[str, SimulationResult]:
    """Fait tourner chaque variante sur la même population et les mêmes marchés.
    Clés d'une variante : name, social (native|oasis|off), news (bool), anonymize (bool),
    mode (rule|llm), oasis_reactions (rule|llm)."""
    from .agents import build_population
    from .data.scenario import build_scenario
    from .sim.runner import run_simulation
    results: Dict[str, SimulationResult] = {}
    for v in variants:
        scenario = build_scenario(series, start, end, bool(v.get('anonymize', False)))
        if not v.get('news', True):
            scenario = without_news(scenario)
        llm = llm_factory() if v.get('mode', 'rule') == 'llm' and llm_factory else None
        results[v['name']] = run_simulation(
            build_population(seed), scenario, llm, data_source, social=v.get('social', 'native'),
            oasis_reactions=v.get('oasis_reactions', 'rule'))
    return results


def compare_variants(results: Mapping[str, SimulationResult], variants: Sequence[dict],
                     reference: Optional[Mapping[str, Mapping]] = None) -> dict:
    names = [v['name'] for v in variants]
    base = results[names[0]]
    notes = []
    if any(v.get('mode', 'rule') == 'rule' and not v.get('news', True) for v in variants) or \
            len({v.get('news', True) for v in variants if v.get('mode', 'rule') == 'rule'}) > 1:
        notes.append(NEWS_NOTE)
    out = {
        'base': names[0],
        'allocations': {n: allocation_report(results[n]) for n in names},
        'comparisons': {n: compare(base, results[n]) for n in names[1:]},
        'notes': notes,
    }
    if reference:
        out['reference'] = {n: compare_to_reference(results[n], reference) for n in names}
    return out
