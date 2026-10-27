import dataclasses
import itertools
import math

import pytest

from app.fundsim.agents import allocation_status, build_population, open_funds
from app.fundsim.data.scenario import build_scenario, synthetic_series_returns
from app.fundsim.sim.decisions import HERDING_GAIN, build_messages, make_candidates, rule_proposal
from app.fundsim.sim.runner import run_simulation
from app.fundsim.sim.social import FOLLOWS_PER_LP, Feed, PeerView, Post, build_graph


def scenario(seed=0):
    return build_scenario(synthetic_series_returns('2019-01', '2023-12', seed), '2019-01', '2023-12')


def test_graph_is_deterministic_and_well_formed():
    pop = build_population(7)
    g1, g2, g3 = build_graph(pop, 7), build_graph(pop, 7), build_graph(pop, 8)
    assert g1.follows == g2.follows and g1.follows != g3.follows
    ids = {a.agent_id for a in pop.lps}
    for lp, followed in g1.follows.items():
        assert len(followed) == FOLLOWS_PER_LP == len(set(followed)) and lp not in followed and set(followed) <= ids
    assert all(v > 0 for v in g1.influence.values())


def test_prestige_and_size_drive_influence():
    pop = build_population(7)
    g = build_graph(pop, 7)
    avg = lambda arch: sum(g.influence[a.agent_id] for a in pop.lps if a.archetype == arch) / \
        sum(1 for a in pop.lps if a.archetype == arch)
    assert avg('canadian_pension') > avg('family_office') and avg('endowment') > avg('corporate_pension')


def test_same_archetype_is_followed_more_often():
    pop = build_population(7)
    arch = {a.agent_id: a.archetype for a in pop.lps}
    same = total = 0
    for seed in range(30):
        for lp, followed in build_graph(pop, seed).follows.items():
            same += sum(arch[f] == arch[lp] for f in followed)
            total += len(followed)
    peers_same = sum(1 for a in pop.lps for b in pop.lps if a is not b and a.archetype == b.archetype)
    baseline = peers_same / (len(pop.lps) * (len(pop.lps) - 1))
    assert same / total > baseline * 1.5


def make_feed():
    pop = build_population(7)
    g = build_graph(pop, 7)
    me = pop.lps[0].agent_id
    followed, stranger = g.follows[me][0], next(a.agent_id for a in pop.lps if a.agent_id not in g.follows[me] + [me])
    return Feed(g), me, followed, stranger


def test_peer_view_only_sees_followed_authors_in_window():
    feed, me, followed, stranger = make_feed()
    feed.publish(Post(10, 'm10', followed, 'F-A', 0.5, 100, "j'y crois"))
    feed.publish(Post(10, 'm10', stranger, 'F-B', 0.5, 100, "inconnu"))
    feed.publish(Post(2, 'm2', followed, 'F-C', 0.5, 100, "trop ancien"))
    v = feed.peer_view(me, 11)
    assert set(v.weights) == {'F-A'} and v.weights['F-A'] == pytest.approx(1.0) and v.n_peers == 1
    assert v.snippets == ["j'y crois"]
    assert feed.peer_view(me, 20).weights == {}          # tout est sorti de la fenêtre
    assert feed.peer_view(stranger, 11).n_peers <= len(feed.graph.follows[stranger])


def test_peer_weights_are_influence_weighted_and_normalized():
    feed, me, _, _ = make_feed()
    a, b = feed.graph.follows[me][:2]
    feed.graph.influence[a], feed.graph.influence[b] = 3.0, 1.0
    feed.publish(Post(5, 'm5', a, 'F-A', 1.0, 10, ""))
    feed.publish(Post(5, 'm5', b, 'F-B', 1.0, 10, ""))
    w = feed.peer_view(me, 5).weights
    assert w['F-A'] == pytest.approx(0.75) and w['F-B'] == pytest.approx(0.25) and sum(w.values()) == pytest.approx(1)


def setup_candidates():
    pop = build_population(7)
    cands = make_candidates(open_funds(pop.funds, 20), pop.gps, {}, {})
    return pop, cands


def test_herding_boosts_crowded_funds_in_proportion_to_trait():
    pop, cands = setup_candidates()
    lp = pop.lps[0]
    target = next(c for c in cands if lp.policy.strategy_targets.get(c.fund.strategy, 0) > 0)
    peers = {target.fund.fund_id: 0.8, 'other': 0.2}
    base = rule_proposal(lp, cands)
    boosted = rule_proposal(lp, cands, peer_weights=peers)
    ratio = boosted[target.fund.fund_id] / base[target.fund.fund_id]
    assert ratio == pytest.approx(1 + HERDING_GAIN * lp.policy.herding)
    independent = dataclasses.replace(lp, policy=dataclasses.replace(lp.policy, herding=0.0))
    assert rule_proposal(independent, cands, peer_weights=peers) == rule_proposal(independent, cands)
    assert rule_proposal(lp, cands, peer_weights={}) == base


def test_perception_noise_is_individual_and_reproducible():
    pop, cands = setup_candidates()
    a, b = pop.lps[0], pop.lps[1]
    ra, rb = rule_proposal(a, cands, noise_key='m'), rule_proposal(b, cands, noise_key='m')
    assert ra == rule_proposal(a, cands, noise_key='m')
    assert ra != rule_proposal(a, cands, noise_key='other-month')
    assert ra != rule_proposal(a, cands) and ra.keys() == rb.keys()


def test_peer_summary_reaches_the_prompt():
    pop, cands = setup_candidates()
    lp = pop.lps[0]
    fid = cands[0].fund.fund_id
    view = PeerView({fid: 1.0}, ["Je renforce mon allocation."], 3)
    text = view.summary(pop.funds, lp.policy.herding)
    assert fid in text and '100%' in text and 'Je renforce' in text and f"{lp.policy.herding:.0%}" in text
    sc = scenario()
    msgs = build_messages(lp, allocation_status(lp.account, lp.policy), cands, sc.months[20], [], text)
    assert 'Ce que font les autres investisseurs' in msgs[1]['content'] and fid in msgs[1]['content']
    empty = build_messages(lp, allocation_status(lp.account, lp.policy), cands, sc.months[20], [], "")
    assert 'autres investisseurs' not in empty[1]['content']
    assert PeerView({}, [], 0).summary(pop.funds, 0.5) == ""


def test_social_off_has_empty_feed_and_social_on_posts_only_real_commitments():
    off = run_simulation(build_population(7), scenario(), None, 'synthetic', social=False)
    assert off.social_enabled is False and off.feed == [] and off.follows == {}
    pop = build_population(7)
    on = run_simulation(pop, scenario(), None, 'synthetic', social=True)
    assert on.social_enabled and on.feed and len(on.feed) <= 200 and on.influence and on.follows
    committed = {fid for lp in pop.lps for fid in lp.account.positions}
    assert all(p['fund_id'] in committed and p['amount'] > 0 for p in on.feed)


def test_social_run_is_deterministic():
    r1 = run_simulation(build_population(7), scenario(), None, 'synthetic').to_dict()
    r2 = run_simulation(build_population(7), scenario(), None, 'synthetic').to_dict()
    assert r1 == r2


def _mean_similarity(seed, social):
    pop = build_population(seed)
    run_simulation(pop, scenario(seed), None, 'synthetic', social=social)
    vec = {lp.agent_id: {f: p.commitment for f, p in lp.account.positions.items() if not f.startswith('LEG-')}
           for lp in pop.lps}

    def cos(a, b):
        na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
        return sum(v * b.get(k, 0) for k, v in a.items()) / (na * nb)
    sims = [cos(vec[x.agent_id], vec[y.agent_id]) for x, y in itertools.combinations(pop.lps, 2)]
    return sum(sims) / len(sims)


def test_imitation_makes_lps_converge_on_the_same_funds():
    gains = [_mean_similarity(s, True) - _mean_similarity(s, False) for s in (3, 7, 11)]
    assert all(g > 0 for g in gains)


def test_dust_commitments_are_ignored():
    from app.fundsim.sim.runner import MIN_COMMITMENT
    pop = build_population(7)
    r = run_simulation(pop, scenario(), None, 'synthetic')
    assert all(p['amount'] >= MIN_COMMITMENT - 0.5 for p in r.feed)
    for lp in pop.lps:
        for fid, pos in lp.account.positions.items():
            assert fid.startswith('LEG-') or pos.commitment >= MIN_COMMITMENT
