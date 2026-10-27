import pytest
from flask import Flask

from app.config import Config
from app.fundsim import fundsim_bp
from app.fundsim.llm import BudgetExceeded, FundSimLLM, LLMJSONError, extract_json


def test_validate_does_not_require_zep(monkeypatch):
    monkeypatch.setattr(Config, 'LLM_API_KEY', 'k')
    monkeypatch.setattr(Config, 'ZEP_API_KEY', None)
    assert Config.validate() == []


def test_validate_requires_llm_key_unless_dry_run(monkeypatch):
    monkeypatch.setattr(Config, 'LLM_API_KEY', None)
    monkeypatch.setattr(Config, 'FUNDSIM_DRY_RUN', False)
    assert Config.validate()
    monkeypatch.setattr(Config, 'FUNDSIM_DRY_RUN', True)
    assert Config.validate() == []


@pytest.mark.parametrize('text', ['{"a": 1}', 'texte\n```json\n{"a": 1}\n```', 'ok {"a": 1} fin'])
def test_extract_json(text):
    assert extract_json(text) == {'a': 1}


@pytest.mark.parametrize('text', ['', 'pas de json', '[1, 2]', '{"a": '])
def test_extract_json_rejects(text):
    with pytest.raises(LLMJSONError):
        extract_json(text)


def test_budget_is_enforced():
    llm = FundSimLLM(max_calls=2, dry_run=False, transport=lambda *a: '{"x": 1}')
    llm.complete([{"role": "user", "content": "a"}])
    llm.complete([{"role": "user", "content": "b"}])
    with pytest.raises(BudgetExceeded):
        llm.complete([{"role": "user", "content": "c"}])
    assert llm.calls == 2 and llm.remaining == 0


def test_json_retry_then_success():
    replies = iter(['désolé', '{"w": 0.5}'])
    llm = FundSimLLM(max_calls=5, dry_run=False, transport=lambda *a: next(replies))
    assert llm.complete_json([{"role": "user", "content": "q"}]) == {'w': 0.5}
    assert llm.calls == 2


def test_json_retry_exhausted():
    llm = FundSimLLM(max_calls=5, dry_run=False, transport=lambda *a: 'nope')
    with pytest.raises(LLMJSONError):
        llm.complete_json([{"role": "user", "content": "q"}], retries=1)


def test_dry_run_makes_no_network_call():
    llm = FundSimLLM(max_calls=3, dry_run=True, transport=lambda *a: pytest.fail('réseau'))
    assert llm.complete_json([{"role": "user", "content": "q"}]) == {'dry_run': True}


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(fundsim_bp, url_prefix='/api/fundsim')
    return app.test_client()


def test_health_exposes_no_secret(client, monkeypatch):
    monkeypatch.setattr(Config, 'LLM_API_KEY', 'sk-secret-value')
    body = client.get('/api/fundsim/health').get_data(as_text=True)
    assert 'sk-secret-value' not in body
    assert '"zep_required":false' in body.replace(' ', '').replace('\n', '')


def test_llm_ping_dry_run(client, monkeypatch):
    monkeypatch.setattr(Config, 'FUNDSIM_DRY_RUN', True)
    res = client.post('/api/fundsim/llm-ping')
    assert res.status_code == 200 and res.get_json()['data']['calls'] == 1
