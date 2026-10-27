import json
import statistics

import pytest
from flask import Flask

from app.config import Config
from app.fundsim import fundsim_bp
from app.fundsim.agents import allocation_status, build_population
from app.fundsim.analysis import (
    NEWS_NOTE, allocation_report, compare, compare_to_reference, compare_variants, flows_by_year, hhi,
    run_variants, trajectory, tv_distance, without_news,
)
from app.fundsim.data.scenario import build_scenario, synthetic_series_returns
from app.fundsim.llm import FundSimLLM
from app.fundsim.sim.decisions import fund_cap
from app.fundsim.sim.runner import run_simulation

SERIES = synthetic_series_returns('2019-01', '2023-12', 7)


def run(social='native', anonymize=False, news=True, seed=7):
    sc = build_scenario(synthetic_series_returns('2019-01', '2023-12', seed), '2019-01', '2023-12', anonymize)
    if not news:
        sc = without_news(sc)
    return run_simulation(build_population(seed), sc, None, 'synthetic', social=social)


def test_tv_distance_properties():
    a, b = {'x': 0.7, 'y': 0.3}, {'x': 0.2, 'y': 0.5, 'z': 0.3}
    assert tv_distance(a, a) == 0 and tv_distance(a, b) == tv_distance(b, a) == pytest.approx(0.5)
    assert tv_distance({'x': 1}, {'y': 1}) == 1 and tv_distance({}, {}) == 0
    assert hhi({'a': 1, 'b': 1}) == 0.5 and hhi({'a': 1}) == 1.0 and hhi({}) == 0.0


def test_report_describes_where_the_money_goes():
    r = run()
    rep = allocation_report(r)
    assert set(rep['archetypes']) == set(r.final_allocations) and len(rep['archetypes']) == 9
    for arch, row in rep['archetypes'].items():
        assert sum(row['strategy']['final'].values()) == pytest.approx(1.0, abs=1e-3)
        assert sum(row['theme']['final'].values()) == pytest.approx(1.0, abs=1e-3)
        assert row['strategy']['gap_to_target'] == tv_distance(row['strategy']['final'], r.target_allocations[arch])
        assert row['private_share']['target'] == r.target_private_share[arch]
    assert rep['meta']['months'] == 60 and rep['meta']['data_source'] == 'synthetic'


def test_private_share_ends_near_policy_target():
    rep = allocation_report(run())
    for arch, row in rep['archetypes'].items():
        final, target = row['private_share']['final'], row['private_share']['target']
        assert abs(final - target) <= 0.12, (arch, final, target)


def test_flows_by_year_match_totals_and_anonymize_labels():
    r = run()
    flows = flows_by_year(r)
    assert list(flows) == ['2019', '2020', '2021', '2022', '2023']
    for dim, total in (('strategy', r.commitments_by_strategy), ('theme', r.commitments_by_theme)):
        for k, v in total.items():
            assert sum(y[dim].get(k, 0) for y in flows.values()) == pytest.approx(v, rel=1e-3, abs=1)
    assert list(flows_by_year(run(anonymize=True))) == ['Y1', 'Y2', 'Y3', 'Y4', 'Y5']


def test_trajectory_covers_every_month():
    r = run()
    t = trajectory(r, 'endowment')
    assert len(t) == 60 and t[0]['month'] == '2019-01'
    assert all(sum(p['weights'].values()) == pytest.approx(1.0, abs=1e-3) for p in t if p['weights'])
    assert trajectory(r, 'endowment', 'theme')[-1]['weights'] == r.final_theme_allocations['endowment']


def test_identical_runs_have_zero_shift_and_comparison_is_symmetric():
    a, b = run(), run()
    c = compare(a, b)
    assert c['overall'] == {'mean_strategy_shift': 0, 'mean_theme_shift': 0, 'gp_shift': 0, 'gp_concentration_delta': 0}
    off = run(social='off')
    assert compare(a, off)['overall']['mean_strategy_shift'] == compare(off, a)['overall']['mean_strategy_shift']


def test_social_changes_allocations_a_little():
    shift = compare(run(social='off'), run(social='native'))['overall']
    assert 0 < shift['mean_strategy_shift'] < 0.1


def test_anonymization_cannot_change_rule_decisions():
    """Régression : le bruit de perception était graîné par le libellé du mois, donc l'anonymisation
    (M1 au lieu de 2019-01) modifiait les allocations en mode règle."""
    c = compare(run(), run(anonymize=True))
    assert c['overall'] == {'mean_strategy_shift': 0, 'mean_theme_shift': 0, 'gp_shift': 0, 'gp_concentration_delta': 0}


def test_news_do_not_move_rule_decisions_and_the_report_says_so():
    variants = [{'name': 'avec', 'social': 'native'}, {'name': 'sans', 'social': 'native', 'news': False}]
    res = run_variants(7, SERIES, '2019-01', '2023-12', variants, 'synthetic')
    out = compare_variants(res, variants)
    assert out['comparisons']['sans']['overall']['mean_strategy_shift'] == 0 and NEWS_NOTE in out['notes']


def test_news_can_move_llm_decisions():
    def transport(model, messages, t, mt):
        prompt = messages[1]['content']
        cands = json.loads(prompt.split('Fonds ouverts aux engagements :\n')[1].split('\n\n')[0])
        has_news = '(rien de notable)' not in prompt
        pick = [c for c in cands if c['strategy'] == ('private_credit' if has_news else 'venture')] or cands
        return json.dumps({"weights": {pick[0]['fund_id']: 1}, "rationale": "x"})
    variants = [{'name': 'avec', 'mode': 'llm', 'social': 'off'}, {'name': 'sans', 'mode': 'llm', 'social': 'off', 'news': False}]
    res = run_variants(7, SERIES, '2019-01', '2023-12', variants, 'synthetic',
                       llm_factory=lambda: FundSimLLM(max_calls=2000, dry_run=False, transport=transport))
    out = compare_variants(res, variants)
    assert out['comparisons']['sans']['overall']['mean_strategy_shift'] > 0.05 and NEWS_NOTE not in out['notes']


def test_without_news_does_not_mutate_the_scenario():
    sc = build_scenario(SERIES, '2019-01', '2023-12')
    n = sum(len(m.news) for m in sc.months)
    assert n > 0 and sum(len(m.news) for m in without_news(sc).months) == 0
    assert sum(len(m.news) for m in sc.months) == n


def test_compare_to_reference():
    r = run()
    final = r.final_allocations['insurer']
    ref = {'insurer': {'strategy': final, 'private_share': r.months[-1].private_share['insurer']},
           'endowment': {'strategy': {'venture': 1.0}}, 'nobody': {'strategy': {'buyout': 1.0}}}
    out = compare_to_reference(r, ref)
    assert out['by_archetype']['insurer'] == {'strategy_gap': 0, 'private_share_gap': 0}
    assert out['by_archetype']['endowment']['strategy_gap'] == pytest.approx(1 - r.final_allocations['endowment']['venture'], abs=1e-3)
    assert out['unknown_archetypes'] == ['nobody']


def test_fund_cap_is_relative_to_the_strategy_sleeve():
    pop = build_population(7)
    pension = next(a for a in pop.lps if a.archetype == 'us_public_pension')
    assert fund_cap(pension, 'buyout') == pytest.approx(0.4 * 0.45)
    assert fund_cap(pension, 'buyout') > pension.policy.max_single_fund
    assert fund_cap(pension, 'venture') == pension.policy.max_single_fund


@pytest.mark.parametrize('seed', [3, 7, 11])
def test_calibration_keeps_allocations_close_to_policy(seed):
    pop = build_population(seed)
    r = run_simulation(pop, build_scenario(synthetic_series_returns('2019-01', '2023-12', seed), '2019-01', '2023-12'),
                       None, 'synthetic', social='native')
    bud = sum(d.budget for m in r.months for d in m.decisions)
    unallocated = sum(d.unallocated * d.budget for m in r.months for d in m.decisions) / bud
    gaps = [row['strategy']['gap_to_target'] for row in allocation_report(r)['archetypes'].values()]
    ratios = [allocation_status(l.account, l.policy).exposure_share / l.policy.target_private for l in pop.lps]
    assert unallocated < 0.15 and statistics.mean(gaps) < 0.12 and 0.85 <= statistics.median(ratios) <= 1.2


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(fundsim_bp, url_prefix='/api/fundsim')
    return app.test_client()


def test_api_run_returns_allocations_and_hides_social_detail(client):
    body = client.post('/api/fundsim/run', json={'data': 'synthetic'}).get_json()['data']
    assert set(body) == {'allocations'} and 'insurer' in body['allocations']['archetypes']
    full = client.post('/api/fundsim/run', json={'data': 'synthetic', 'detail': True}).get_json()['data']
    assert 'feed' in full['detail'] and full['allocations'] == body['allocations']


def test_api_compare(client):
    variants = [{'name': 'sans', 'social': 'off'}, {'name': 'avec', 'social': 'native'}]
    res = client.post('/api/fundsim/compare', json={'data': 'synthetic', 'variants': variants,
                                                    'reference': {'insurer': {'private_share': 0.1}}})
    data = res.get_json()['data']
    assert res.status_code == 200 and data['base'] == 'sans' and set(data['allocations']) == {'sans', 'avec'}
    assert 'avec' in data['comparisons'] and set(data['reference']) == {'sans', 'avec'}


@pytest.mark.parametrize('variants', [
    [{'name': 'a'}], [{'name': 'a'}, {'name': 'a'}], [{'name': 'a'}, {'name': 'b', 'social': 'zzz'}],
    [{'name': 'a'}, {}], [{'name': 'a'}, {'name': 'b', 'mode': 'zzz'}],
])
def test_api_compare_validates_variants(client, variants):
    assert client.post('/api/fundsim/compare', json={'data': 'synthetic', 'variants': variants}).status_code == 400


def test_api_compare_needs_llm_key_for_llm_variants(client, monkeypatch):
    monkeypatch.setattr(Config, 'LLM_API_KEY', None)
    monkeypatch.setattr(Config, 'FUNDSIM_DRY_RUN', False)
    res = client.post('/api/fundsim/compare', json={'data': 'synthetic', 'variants': [{'name': 'a'}, {'name': 'b', 'mode': 'llm'}]})
    assert res.status_code == 400
