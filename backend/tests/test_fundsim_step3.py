import pytest
from flask import Flask

from app.fundsim import fundsim_bp
from app.fundsim.agents import (
    ARCHETYPES, DEFAULT_COUNTS, GPS, allocation_status, build_population, build_universe,
    fund_return, is_review_month, open_funds, prompt_block,
)
from app.fundsim.engine import STRATEGIES, LPAccount, validate_allocation


@pytest.mark.parametrize('arch', sorted(ARCHETYPES))
def test_archetype_is_coherent(arch):
    p = ARCHETYPES[arch]['policy']
    lo, hi = p.band
    assert lo <= p.target_private <= hi
    assert sum(p.strategy_targets.values()) == pytest.approx(1.0)
    assert set(p.strategy_targets) == set(STRATEGIES)
    assert 0 < p.pacing_annual <= 0.3 and 0 <= p.herding <= 1
    assert p.limits().max_single_fund == p.max_single_fund


def test_population_is_reproducible_and_seed_sensitive():
    a, b, c = build_population(7), build_population(7), build_population(8)
    assert a.summary() == b.summary()
    assert a.summary()['lps'] != c.summary()['lps']
    assert len(a.lps) == sum(DEFAULT_COUNTS.values())
    assert len({x.name for x in a.lps}) == len(a.lps)


def test_initial_book_matches_policy():
    for lp in build_population(7).lps:
        exposure = (lp.account.private_nav) / lp.account.total_assets
        assert exposure == pytest.approx(lp.policy.target_private * 0.8, rel=1e-6)
        assert allocation_status(lp.account, lp.policy).state in ('under', 'in_band')


def test_every_position_has_a_known_fund():
    pop = build_population(7)
    for lp in pop.lps:
        assert set(lp.account.positions) <= set(pop.funds)


def test_universe_and_open_window():
    funds, gps = build_universe(60)
    assert all(f.gp in gps for f in funds.values())
    assert {f.strategy for f in funds.values()} == set(STRATEGIES)
    assert len(open_funds(funds, 0)) >= 8       # le marché est déjà ouvert au premier mois
    assert len(open_funds(funds, 20)) >= 6     # assez de choix pour diversifier
    assert open_funds(funds, 13) != []
    assert all(f.vintage_month <= 13 < f.vintage_month + 18 for f in open_funds(funds, 13))


def test_fund_returns_deterministic_and_skill_matters():
    funds, gps = build_universe(60)
    f = funds['lumen-1']
    r1 = fund_return(f, gps['lumen'], 0.01, 5, seed=7)
    assert r1 == fund_return(f, gps['lumen'], 0.01, 5, seed=7)
    assert r1 != fund_return(f, gps['lumen'], 0.01, 5, seed=8)
    # sur 1 200 tirages, un meilleur alpha net donne un rendement moyen plus élevé
    good = sum(fund_return(f, gps['lumen'], 0.0, m, 1) for m in range(1200)) / 1200
    bad = sum(fund_return(funds['helix-1'], gps['helix'], 0.0, m, 1) for m in range(1200)) / 1200
    assert good > bad


def test_review_months_are_staggered():
    for k in range(5):
        reviews = [is_review_month(k, m, 3) for m in range(12)]
        assert sum(reviews) == 4
    assert {i for i in range(3) if is_review_month(i, 0, 3)} == {0}
    per_month = [sum(is_review_month(k, m, 3) for k in range(9)) for m in range(3)]
    assert per_month == [3, 3, 3]


def test_denominator_effect_blocks_new_commitments():
    pop = build_population(7)
    lp = pop.lps[0]
    before = allocation_status(lp.account, lp.policy)
    assert before.budget > 0 and before.reason == 'ok'
    lp.account.liquid *= 0.3                       # les actifs cotés s'effondrent
    after = allocation_status(lp.account, lp.policy)
    assert after.state == 'over' and after.budget == 0 and 'dénominateur' in after.reason


def test_liquidity_floor_blocks_commitments():
    pop = build_population(7)
    lp = next(a for a in pop.lps if a.archetype == 'insurer')
    lp.account.liquid = lp.account.private_nav * 0.05 / 0.95    # 5 % de liquidités
    st = allocation_status(lp.account, lp.policy)
    assert st.budget == 0 and 'liquidités' in st.reason


def test_budget_never_exceeds_room_to_ceiling():
    for lp in build_population(7).lps:
        st = allocation_status(lp.account, lp.policy)
        room = lp.policy.band[1] * lp.account.total_assets - (lp.account.private_nav + lp.account.unfunded)
        assert st.budget <= max(room, 0) + 1e-6


def test_empty_account_has_no_budget():
    lp = build_population(7).lps[0]
    assert allocation_status(LPAccount('X', 0.0), lp.policy).budget == 0


def test_prompt_block_contains_mandate():
    lp = build_population(7).lps[0]
    text = prompt_block(lp.name, lp.archetype, lp.policy, allocation_status(lp.account, lp.policy))
    assert lp.name in text and 'Budget d\'engagement' in text and 'buyout' in text


def test_theme_limits_plug_into_validation():
    pop = build_population(7)
    lp = pop.lps[0]
    candidates = {f.fund_id: f for f in open_funds(pop.funds, 12)}
    themes = {f.theme for f in candidates.values()}
    res = validate_allocation({fid: 1 for fid in candidates}, candidates, lp.policy.theme_limits(themes))
    assert sum(res.weights.values()) + res.unallocated == pytest.approx(1.0)
    assert all(w <= lp.policy.max_single_fund + 1e-9 for w in res.weights.values())


def test_population_endpoint():
    app = Flask(__name__)
    app.register_blueprint(fundsim_bp, url_prefix='/api/fundsim')
    data = app.test_client().get('/api/fundsim/population?seed=3').get_json()['data']
    assert data['seed'] == 3 and len(data['lps']) == 17 and len(data['gps']) == len(GPS)


def test_unfunded_commitments_count_against_liquidity():
    lp = next(a for a in build_population(7).lps if a.archetype == 'family_office')
    assert allocation_status(lp.account, lp.policy).reason == 'ok'
    lp.account.positions['LEG-venture'].unfunded = lp.account.liquid * 0.95   # presque tout le cash est déjà promis
    st = allocation_status(lp.account, lp.policy)
    assert st.budget == 0 and 'après appels' in st.reason
