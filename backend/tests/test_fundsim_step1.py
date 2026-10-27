import math

import pytest

from app.fundsim.engine import (
    Fund, FundParams, Limits, LPAccount, fundraising_shares, monthly_rate,
    trailing_score, validate_allocation,
)


def make_funds():
    rows = [
        ('F1', 'GP1', 'buyout', 'tech'), ('F2', 'GP2', 'buyout', 'health'),
        ('F3', 'GP3', 'venture', 'tech'), ('F4', 'GP4', 'private_credit', 'energy'),
        ('F5', 'GP5', 'infrastructure', 'energy'),
    ]
    return {r[0]: Fund(r[0], r[1], r[2], r[3], 0, 1000.0) for r in rows}


def run(lp, funds, months, ret=0.0, public=0.0):
    log = []
    for _ in range(months):
        log.append(lp.step_month(funds, {fid: ret for fid in lp.positions}, public))
    return log


def test_monthly_rate_compounds_back_to_annual():
    for a in (0.0, 0.1, 0.25, 0.9):
        assert (1 - monthly_rate(a)) ** 12 == pytest.approx(1 - a)
    assert monthly_rate(1.0) == 1.0 and monthly_rate(-1) == 0.0


def test_calls_never_exceed_commitment_and_unfunded_expires():
    funds = make_funds()
    lp = LPAccount('LP', liquid=10_000)
    lp.commit('F1', 1000)
    run(lp, funds, 12 * 5)
    pos = lp.positions['F1']
    assert pos.called <= 1000 + 1e-6
    assert pos.unfunded == 0.0


def test_cash_conservation_with_zero_returns():
    funds = make_funds()
    lp = LPAccount('LP', liquid=10_000)
    lp.commit('F1', 1000)
    log = run(lp, funds, 12 * 12)
    net_cash = sum(m.distributions - m.calls for m in log)
    assert lp.liquid == pytest.approx(10_000 + net_cash)
    pos = lp.positions['F1']
    assert pos.called - pos.distributed == pytest.approx(pos.nav)


def test_fund_fully_distributed_at_end_of_life():
    funds = make_funds()
    lp = LPAccount('LP', liquid=10_000)
    lp.commit('F1', 1000)
    run(lp, funds, 12 * 11)
    assert lp.positions['F1'].nav == pytest.approx(0.0, abs=1e-6)


def test_j_curve_negative_then_positive():
    funds = make_funds()
    lp = LPAccount('LP', liquid=10_000)
    lp.commit('F1', 1000)
    cum, lows = 0.0, []
    for m in run(lp, funds, 12 * 11, ret=0.008):
        cum += m.distributions - m.calls
        lows.append(cum)
    assert min(lows) < 0 < lows[-1]
    assert lp.positions['F1'].tvpi > 1.0


def test_unmet_calls_flagged_when_liquidity_runs_out():
    funds = make_funds()
    lp = LPAccount('LP', liquid=10)
    lp.commit('F1', 1000)
    run(lp, funds, 24)
    assert lp.unmet_calls > 0 and lp.liquid >= 0


def test_exposure_sums_to_one():
    funds = make_funds()
    lp = LPAccount('LP', liquid=0)
    lp.commit('F1', 300)
    lp.commit('F3', 100)
    exp = lp.exposure(funds, 'strategy')
    assert exp == {'buyout': 0.75, 'venture': 0.25}


LIMITS = Limits(max_by_strategy={'buyout': 0.5}, max_by_theme={'tech': 0.4}, max_single_fund=0.3)


def check_caps(res, funds, limits=LIMITS):
    strat, theme = {}, {}
    for fid, w in res.weights.items():
        assert w <= limits.max_single_fund + 1e-9
        strat[funds[fid].strategy] = strat.get(funds[fid].strategy, 0) + w
        theme[funds[fid].theme] = theme.get(funds[fid].theme, 0) + w
    for k, cap in limits.max_by_strategy.items():
        assert strat.get(k, 0) <= cap + 1e-9
    for k, cap in limits.max_by_theme.items():
        assert theme.get(k, 0) <= cap + 1e-9


def test_valid_allocation_is_untouched():
    funds = make_funds()
    res = validate_allocation({'F1': 0.2, 'F2': 0.2, 'F3': 0.2, 'F4': 0.2, 'F5': 0.2}, funds, LIMITS)
    check_caps(res, funds)
    assert sum(res.weights.values()) == pytest.approx(1.0) and res.unallocated == pytest.approx(0.0)


def test_caps_are_enforced_and_excess_redistributed():
    funds = make_funds()
    res = validate_allocation({'F1': 0.9, 'F2': 0.05, 'F4': 0.05}, funds, LIMITS)
    check_caps(res, funds)
    assert res.weights['F1'] <= 0.3 + 1e-9
    assert sum(res.weights.values()) + res.unallocated == pytest.approx(1.0)
    assert any('F1' in a for a in res.adjustments)


def test_infeasible_caps_leave_unallocated():
    funds = make_funds()
    tight = Limits(max_single_fund=0.1)
    res = validate_allocation({'F1': 1, 'F2': 1}, funds, tight)
    assert sum(res.weights.values()) == pytest.approx(0.2)
    assert res.unallocated == pytest.approx(0.8)


def test_bad_inputs_are_dropped():
    funds = make_funds()
    res = validate_allocation({'F1': 1, 'GHOST': 5, 'F2': -1, 'F3': float('nan'), 'F4': 'x'}, funds,
                              Limits(max_single_fund=1.0))
    assert set(res.weights) == {'F1'}
    assert len(res.adjustments) == 4


def test_empty_proposal_is_fully_unallocated():
    res = validate_allocation({}, make_funds(), LIMITS)
    assert res.weights == {} and res.unallocated == 1.0


def test_fund_specific_capacity_cap():
    funds = make_funds()
    res = validate_allocation({'F1': 0.5, 'F2': 0.5}, funds, Limits(max_single_fund=1.0),
                              fund_caps={'F1': 0.1})
    assert res.weights['F1'] == pytest.approx(0.1)


def test_validation_is_deterministic():
    funds = make_funds()
    p = {'F1': 0.4, 'F2': 0.3, 'F3': 0.2, 'F4': 0.1}
    assert validate_allocation(p, funds, LIMITS).weights == validate_allocation(p, funds, LIMITS).weights


def test_trailing_score_and_fundraising():
    assert trailing_score([0.1, 0.1]) == pytest.approx(0.21)
    assert trailing_score([0.5, 0.0, 0.0], window=2) == pytest.approx(0.0)
    shares = fundraising_shares({'A': 0.20, 'B': 0.05, 'C': -0.10}, temperature=0.05)
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares['A'] > shares['B'] > shares['C']
    flat = fundraising_shares({'A': 0.20, 'B': 0.05}, temperature=1000)
    assert abs(flat['A'] - 0.5) < 0.01
    assert fundraising_shares({}) == {}
