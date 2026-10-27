"""Couche sociale adossée à OASIS. Les tests qui font tourner OASIS sont ignorés si le paquet est absent
(installation légère : backend/requirements-oasis-lite.txt)."""

import dataclasses
import os
import sqlite3

import pytest
from flask import Flask

from app.config import Config
from app.fundsim import fundsim_bp
from app.fundsim.agents import build_population
from app.fundsim.data.scenario import build_scenario, synthetic_series_returns
from app.fundsim.oasis_compat import OasisUnavailable, ensure_oasis, oasis_available
from app.fundsim.sim.runner import run_simulation
from app.fundsim.sim.social import ENDORSE_GAIN, Feed, Post, build_graph

needs_oasis = pytest.mark.skipif(not oasis_available(), reason="OASIS non installé")
SMALL = {'us_public_pension': 2, 'endowment': 1, 'family_office': 2, 'insurer': 1}


def small_run(tmp_path=None, seed=7, population=None, **kw):
    pop = population or build_population(seed, counts=SMALL)
    sc = build_scenario(synthetic_series_returns('2019-01', '2020-12', seed), '2019-01', '2020-12')
    return pop, run_simulation(pop, sc, None, 'synthetic', social='oasis',
                               oasis_db_dir=str(tmp_path) if tmp_path else None, **kw)


def test_endorsements_boost_peer_weights():
    pop = build_population(7)
    g = build_graph(pop, 7)
    me = pop.lps[0].agent_id
    a, b = g.follows[me][:2]
    g.influence[a] = g.influence[b] = 1.0
    feed = Feed(g)
    feed.publish(Post(5, 'm5', a, 'F-A', 0.5, 10, ""))
    feed.publish(Post(5, 'm5', b, 'F-B', 0.5, 10, ""))
    assert feed.peer_view(me, 5).weights['F-A'] == pytest.approx(0.5)
    feed.endorsements[(a, 'F-A', 5)] = 4
    w = feed.peer_view(me, 5).weights
    assert w['F-A'] / w['F-B'] == pytest.approx(1 + ENDORSE_GAIN * 4)
    feed.comments[(a, 'F-A', 5)] = ["Très bon GP."]
    assert "Très bon GP." in feed.peer_view(me, 5).snippets


def test_missing_oasis_gives_actionable_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == 'oasis':
            raise ImportError("No module named 'oasis'")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', fake)
    with pytest.raises(OasisUnavailable, match='requirements-oasis-lite'):
        ensure_oasis()


def test_stubs_never_shadow_a_real_torch():
    import sys
    from app.fundsim import oasis_compat
    ensure = oasis_compat.STUBS_DIR
    ensure_oasis_path = [p for p in sys.path if p == ensure]
    if ensure_oasis_path:
        assert sys.path.index(ensure) > 0 and sys.path[0] != ensure     # jamais en tête


@needs_oasis
def test_oasis_run_creates_real_platform_content(tmp_path):
    pop, r = small_run(tmp_path)
    con = sqlite3.connect(str(tmp_path / 'oasis.db'))
    posts = con.execute("select count(*) from post").fetchone()[0]
    likes = con.execute("select count(*) from like").fetchone()[0]
    follows = con.execute("select count(*) from follow").fetchone()[0]
    assert r.social_backend == 'oasis' and r.oasis['mode'] == 'rule'
    assert posts == r.oasis['posts'] > 0 and likes == r.oasis['likes'] and follows == sum(len(v) for v in r.follows.values())
    assert sum(r.decision_sources.values()) > 0 and sum(r.commitments_by_gp.values()) > 0


@needs_oasis
def test_only_followers_react(tmp_path):
    pop, r = small_run(tmp_path)
    con = sqlite3.connect(str(tmp_path / 'oasis.db'))
    ids = {a.index: a.agent_id for a in pop.lps}
    for liker, author in con.execute("select l.user_id, p.user_id from like l join post p on p.post_id = l.post_id"):
        assert ids[author] in r.follows[ids[liker]]


@needs_oasis
def test_independent_lps_never_react(tmp_path):
    pop = build_population(7, counts=SMALL)
    for lp in pop.lps:
        lp.policy = dataclasses.replace(lp.policy, herding=0.0)
    _, r = small_run(population=pop)
    assert r.oasis['likes'] == 0 and r.oasis['comments'] == 0 and r.oasis['posts'] > 0


@needs_oasis
def test_oasis_run_is_deterministic():
    _, a = small_run()
    _, b = small_run()
    assert a.to_dict() == b.to_dict()


@needs_oasis
def test_reactions_change_what_peers_see():
    _, oasis_r = small_run()
    pop = build_population(7, counts=SMALL)
    sc = build_scenario(synthetic_series_returns('2019-01', '2020-12', 7), '2019-01', '2020-12')
    native = run_simulation(pop, sc, None, 'synthetic', social='native')
    assert oasis_r.oasis['likes'] > 0 and native.oasis == {}
    assert oasis_r.commitments_by_gp != native.commitments_by_gp     # les réactions modifient les décisions


@needs_oasis
def test_temp_database_is_cleaned_up():
    import glob
    import tempfile
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), 'fundsim-oasis-*')))
    small_run()
    assert set(glob.glob(os.path.join(tempfile.gettempdir(), 'fundsim-oasis-*'))) == before


@needs_oasis
def test_llm_reactions_need_a_key(monkeypatch):
    monkeypatch.setattr(Config, 'LLM_API_KEY', None)
    with pytest.raises(RuntimeError, match='LLM_API_KEY'):
        small_run(oasis_reactions='llm')


@needs_oasis
def test_llm_reactions_respect_their_budget(monkeypatch):
    from app.fundsim.sim import oasis_social
    monkeypatch.setattr(Config, 'LLM_API_KEY', 'fake')
    orig = oasis_social.OasisSession._model

    def stub_model(self):          # modèle factice d'OASIS : ne contacte aucun service
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType, ModelType
        return ModelFactory.create(model_platform=ModelPlatformType.OPENAI, model_type=ModelType.STUB)
    monkeypatch.setattr(oasis_social.OasisSession, '_model', stub_model)
    _, r = small_run(oasis_reactions='llm', oasis_llm_budget=5)
    assert r.oasis['mode'] == 'llm' and r.oasis['llm_actions'] == 5
    monkeypatch.setattr(oasis_social.OasisSession, '_model', orig)


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(fundsim_bp, url_prefix='/api/fundsim')
    return app.test_client()


def test_api_reports_missing_oasis(client, monkeypatch):
    from app.fundsim import oasis_compat

    def boom():
        raise OasisUnavailable(oasis_compat.INSTALL_HINT)
    monkeypatch.setattr('app.fundsim.sim.oasis_social.ensure_oasis', boom)
    res = client.post('/api/fundsim/run', json={'data': 'synthetic', 'social': 'oasis'})
    assert res.status_code == 503 and 'requirements-oasis-lite' in res.get_json()['error']


def test_api_rejects_unknown_social_backend(client):
    assert client.post('/api/fundsim/run', json={'data': 'synthetic', 'social': 'x'}).status_code == 400
    assert client.post('/api/fundsim/run', json={'data': 'synthetic', 'oasis_reactions': 'x'}).status_code == 400
