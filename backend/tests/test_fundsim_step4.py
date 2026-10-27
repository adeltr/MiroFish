import json

import pytest
from flask import Flask

from app.config import Config
from app.fundsim import fundsim_bp
from app.fundsim.agents import allocation_status, build_population, open_funds
from app.fundsim.data.scenario import build_scenario, synthetic_series_returns
from app.fundsim.llm import FundSimLLM
from app.fundsim.sim.decisions import build_messages, decide, fund_cap, make_candidates, rule_proposal
from app.fundsim.sim.runner import run_simulation


def scenario(anon=False, seed=0):
    return build_scenario(synthetic_series_returns('2019-01', '2023-12', seed), '2019-01', '2023-12', anon)


def first_review_setup(month=12):
    pop = build_population(7)
    lp = pop.lps[0]
    cands = make_candidates(open_funds(pop.funds, month), pop.gps, {}, {})
    return pop, lp, cands, allocation_status(lp.account, lp.policy)


def test_rule_proposal_follows_targets_and_tilts():
    pop, lp, cands, _ = first_review_setup()
    raw = rule_proposal(lp, cands)
    assert raw and all(w > 0 for w in raw.values())
    assert rule_proposal(lp, cands) == raw
    fo = next(a for a in pop.lps if a.archetype == 'family_office')
    tech = [c for c in cands if c.fund.theme == 'tech']
    if tech:   # préférence thématique x1,3 pour la technologie
        assert rule_proposal(fo, tech)[tech[0].fund.fund_id] > 0


def test_unknown_funds_and_overweights_are_clipped_and_logged():
    pop, lp, cands, st = first_review_setup()
    ids = [c.fund.fund_id for c in cands]
    llm = FundSimLLM(max_calls=5, dry_run=False, transport=lambda *a: json.dumps(
        {"weights": {ids[0]: 0.9, "GHOST-9": 0.1}, "rationale": "test"}))
    dec = decide(lp, st, cands, scenario().months[12], [], llm, ['tech'])
    assert dec.source == 'llm' and dec.rationale == 'test'
    assert all(w <= fund_cap(lp, next(c.fund.strategy for c in cands if c.fund.fund_id == fid)) + 1e-9
               for fid, w in dec.weights.items())
    assert any('GHOST-9' in a for a in dec.adjustments)


@pytest.mark.parametrize('transport, source', [
    (lambda *a: 'pas du json', 'fallback:json'),
    (lambda *a: '{"weights": {}}', 'fallback:json'),
])
def test_invalid_llm_answers_fall_back_to_rule(transport, source):
    pop, lp, cands, st = first_review_setup()
    llm = FundSimLLM(max_calls=10, dry_run=False, transport=transport)
    dec = decide(lp, st, cands, scenario().months[12], [], llm, ['tech'])
    assert dec.source == source and dec.weights


def test_budget_exhaustion_falls_back_to_rule():
    pop, lp, cands, st = first_review_setup()
    llm = FundSimLLM(max_calls=0, dry_run=False, transport=lambda *a: pytest.fail('appel'))
    assert decide(lp, st, cands, scenario().months[12], [], llm, ['tech']).source == 'fallback:budget'


def test_prompt_has_news_and_hides_dates_when_anonymized():
    pop, lp, cands, st = first_review_setup(14)
    plain, anon = scenario(False), scenario(True)
    news = [dict(n, month=plain.months[14].label) for n in plain.months[14].news]
    anon_news = [dict(n, month=anon.months[14].label) for n in anon.months[14].news]
    p_text = build_messages(lp, st, cands, plain.months[14], news)[1]['content']
    a_text = build_messages(lp, st, cands, anon.months[14], anon_news)[1]['content']
    assert '2020-03' in p_text and 'pandémie' in p_text
    assert 'M15' in a_text and '2020' not in a_text and '2019' not in a_text
    assert lp.name in a_text and 'fund_id' in a_text


def test_rule_run_is_deterministic_and_consistent():
    r1 = run_simulation(build_population(7), scenario(), None, 'synthetic')
    r2 = run_simulation(build_population(7), scenario(), None, 'synthetic')
    assert json.dumps(r1.to_dict(), sort_keys=True) == json.dumps(r2.to_dict(), sort_keys=True)
    assert len(r1.months) == 60 and r1.llm_calls == 0 and r1.mode == 'rule'
    total = sum(r1.commitments_by_gp.values())
    assert total > 0
    assert total == pytest.approx(sum(r1.commitments_by_strategy.values()))
    assert total == pytest.approx(sum(r1.commitments_by_theme.values()))
    assert r1.decision_sources == {'rule': sum(r1.decision_sources.values())}
    other = run_simulation(build_population(8), scenario(seed=8), None, 'synthetic')
    assert other.commitments_by_gp != r1.commitments_by_gp


def test_fund_capacity_is_never_exceeded():
    pop = build_population(7)
    run_simulation(pop, scenario(), None, 'synthetic')
    raised = {}
    for lp in pop.lps:
        for fid, pos in lp.account.positions.items():
            if not fid.startswith('LEG-'):
                raised[fid] = raised.get(fid, 0) + pos.commitment
    assert raised and all(v <= pop.funds[k].target_size + 1e-6 for k, v in raised.items())


def test_no_lp_breaks_its_single_fund_cap():
    r = run_simulation(build_population(7), scenario(), None, 'synthetic')
    pop = build_population(7)
    lps = {a.agent_id: a for a in pop.lps}
    for m in r.months:
        for d in m.decisions:
            assert all(w <= fund_cap(lps[d.lp_id], pop.funds[fid].strategy) + 1e-9 for fid, w in d.weights.items())
            assert sum(d.weights.values()) + d.unallocated == pytest.approx(1.0)


def test_llm_run_respects_call_budget_and_logs_sources():
    def transport(model, messages, t, mt):
        cands = json.loads(messages[1]['content'].split('Fonds ouverts aux engagements :\n')[1].split('\n\n')[0])
        return json.dumps({"weights": {c['fund_id']: 1 for c in cands[:2]}, "rationale": "ok"})
    llm = FundSimLLM(max_calls=25, dry_run=False, transport=transport)
    r = run_simulation(build_population(7), scenario(), llm, 'synthetic')
    assert r.llm_calls == 25 and r.mode == 'llm'
    assert r.decision_sources.get('llm') == 25 and r.decision_sources.get('fallback:budget', 0) > 0


def test_dry_run_llm_degrades_to_rule_without_network():
    llm = FundSimLLM(max_calls=1000, dry_run=True, transport=lambda *a: pytest.fail('réseau'))
    r = run_simulation(build_population(7), scenario(), llm, 'synthetic')
    assert set(r.decision_sources) == {'fallback:json'} and sum(r.commitments_by_gp.values()) > 0


def test_final_allocations_sum_to_one_per_archetype():
    r = run_simulation(build_population(7), scenario(), None, 'synthetic')
    for arch, alloc in r.final_allocations.items():
        assert sum(alloc.values()) == pytest.approx(1.0, abs=1e-3)
        assert arch in r.target_allocations


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(fundsim_bp, url_prefix='/api/fundsim')
    return app.test_client()


def test_run_endpoint_synthetic(client):
    res = client.post('/api/fundsim/run', json={'data': 'synthetic', 'mode': 'rule', 'seed': 7})
    body = res.get_json()
    meta = body['data']['allocations']['meta']
    assert res.status_code == 200 and meta['data_source'] == 'synthetic' and meta['months'] == 60
    assert 'detail' not in body['data']          # pas de bruit social par défaut


def test_run_endpoint_without_real_data_explains(client, monkeypatch, tmp_path):
    monkeypatch.setattr(Config, 'FUNDSIM_DATA_DIR', str(tmp_path))
    res = client.post('/api/fundsim/run', json={'data': 'cache'})
    assert res.status_code == 503 and res.get_json()['host_to_allow'] == 'fred.stlouisfed.org'


def test_run_endpoint_validates_input_and_key(client, monkeypatch):
    assert client.post('/api/fundsim/run', json={'mode': 'x'}).status_code == 400
    monkeypatch.setattr(Config, 'LLM_API_KEY', None)
    monkeypatch.setattr(Config, 'FUNDSIM_DRY_RUN', False)
    assert client.post('/api/fundsim/run', json={'mode': 'llm', 'data': 'synthetic'}).status_code == 400


def test_supply_is_calibrated_to_demand():
    pop = build_population(7)
    years = 5.0
    for strat in ('buyout', 'private_credit', 'venture'):
        supply = sum(f.target_size for f in pop.funds.values() if f.gp != 'legacy' and f.strategy == strat)
        demand = sum(max(lp.policy.pacing_annual, 0.25 * lp.policy.target_private) * lp.account.total_assets
                     * years * lp.policy.strategy_targets[strat] for lp in pop.lps)
        assert supply == pytest.approx(1.5 * demand, rel=1e-6)


@pytest.mark.parametrize('seed', [3, 7, 11])
def test_lps_end_near_their_targets_and_funds_do_not_all_fill(seed):
    import statistics
    pop = build_population(seed)
    r = run_simulation(pop, scenario(seed=seed), None, 'synthetic')
    ratios = [allocation_status(lp.account, lp.policy).exposure_share / lp.policy.target_private for lp in pop.lps]
    assert 0.8 <= statistics.median(ratios) <= 1.25
    cap = {}
    for f in pop.funds.values():
        if f.gp != 'legacy':
            cap[f.gp] = cap.get(f.gp, 0) + f.target_size
    fill = [r.commitments_by_gp.get(g, 0) / c for g, c in cap.items()]
    assert max(fill) - min(fill) > 0.3        # la concurrence produit des gagnants et des perdants
    assert statistics.mean(fill) < 0.85       # l'offre n'est pas saturée


def test_gp_prior_history_is_a_skill_signal():
    from app.fundsim.agents.gps import GPS, prior_history
    gps = {g.gp_id: g for g in GPS}
    prior = prior_history(gps, 7)
    assert prior == prior_history(gps, 7) and len(prior['lumen-0']) == 36
    # en moyenne sur beaucoup de graines, un meilleur alpha net donne un meilleur historique
    gaps = [sum(prior_history(gps, s)['lumen-0']) - sum(prior_history(gps, s)['helix-0']) for s in range(40)]
    assert sum(gaps) > 0


def test_under_target_lp_gets_gap_closing_budget():
    lp = build_population(7).lps[0]
    st = allocation_status(lp.account, lp.policy)
    pace = lp.policy.pacing_annual * lp.account.total_assets * lp.policy.review_every / 12
    assert st.budget > pace       # il est sous sa cible : il sur-engage
