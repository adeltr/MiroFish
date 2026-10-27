"""Les séries ci-dessous sont des JEUX FACTICES DE TEST : aucune donnée de marché réelle."""

import re
import urllib.error

import pytest
from flask import Flask

from app.config import Config
from app.fundsim import fundsim_bp
from app.fundsim.data import fred, monthly, news, proxies
from app.fundsim.data.fred import DataError, parse_fred_csv
from app.fundsim.data.scenario import build_scenario, load_series_returns

CSV = "observation_date,SP500\n2023-01-02,.\n2023-01-03,100\n2023-01-31,110\n2023-02-28,99\n"


def write_fake_cache(base, growth=0.01):
    d = base / 'fred'
    d.mkdir(parents=True, exist_ok=True)
    for i, sid in enumerate(proxies.SERIES.values()):
        g = growth * (i + 1) / 3
        rows = ["observation_date,X"]
        level = 100.0
        for y in range(2018, 2024):
            for m in range(1, 13):
                if (y, m) < (2018, 12):
                    continue
                level *= 1 + g * (1 if (y + m) % 2 else -0.5)
                rows.append(f"{y}-{m:02d}-15,{level}")
                rows.append(f"{y}-{m:02d}-28,{level * 1.001}")
        (d / f"{sid}.csv").write_text("\n".join(rows))
    return str(d)


def test_parse_skips_missing_values():
    pts = parse_fred_csv(CSV)
    assert [v for _, v in pts] == [100.0, 110.0, 99.0]


@pytest.mark.parametrize('bad', ['', 'juste une ligne', 'a,b\n2023-01-01,abc'])
def test_parse_rejects_garbage(bad):
    with pytest.raises(DataError):
        parse_fred_csv(bad)


def test_monthly_returns():
    r = monthly.monthly_returns(parse_fred_csv(CSV))
    assert r == {'2023-02': pytest.approx(99 / 110 - 1)}   # janvier n'a pas de mois précédent
    assert monthly.month_range('2022-11', '2023-02') == ['2022-11', '2022-12', '2023-01', '2023-02']


def test_cache_used_before_network(tmp_path, monkeypatch):
    (tmp_path / 'SP500.csv').write_text(CSV)
    monkeypatch.setattr(fred.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('réseau'))
    assert len(fred.load_series('SP500', cache_dir=str(tmp_path))) == 3


def test_network_failure_gives_actionable_error(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError('403')
    monkeypatch.setattr(fred.urllib.request, 'urlopen', boom)
    with pytest.raises(DataError, match='fred.stlouisfed.org'):
        fred.load_series('SP500', cache_dir=str(tmp_path))
    assert not (tmp_path / 'SP500.csv').exists()


def test_offline_mode_without_cache(tmp_path):
    with pytest.raises(DataError):
        fred.load_series('SP500', cache_dir=str(tmp_path), allow_network=False)


def test_smoothing_follows_geltner_recursion():
    months = ['2023-01', '2023-02', '2023-03']
    sr = {k: {'2023-01': 0.10, '2023-02': 0.0, '2023-03': 0.0} for k in proxies.SERIES}
    spec = proxies.ProxySpec({'sp500': 1.0}, 0.0, 0.5)
    out = proxies.strategy_returns(sr, months, {'x': spec})['x']
    assert out['2023-01'] == pytest.approx(0.10)
    assert out['2023-02'] == pytest.approx(0.05)
    assert out['2023-03'] == pytest.approx(0.025)


def test_scenario_from_cache(tmp_path):
    cache = write_fake_cache(tmp_path)
    sr = load_series_returns(cache_dir=cache, allow_network=False)
    sc = build_scenario(sr, '2019-01', '2023-12')
    assert len(sc.months) == 60 and sc.months[0].label == '2019-01'
    assert set(sc.months[0].strategy_returns) == set(proxies.PROXIES)
    assert any(m.news for m in sc.months)
    anon = build_scenario(sr, '2019-01', '2023-12', anonymize=True)
    assert anon.months[0].label == 'M1' and anon.months[-1].label == 'M60'


def test_scenario_reports_missing_months(tmp_path):
    cache = write_fake_cache(tmp_path)
    sr = load_series_returns(cache_dir=cache, allow_network=False)
    sr['sp500'].pop('2020-05')
    with pytest.raises(DataError, match='manquant'):
        build_scenario(sr, '2019-01', '2023-12')


FORBIDDEN = ['Fed', 'Réserve fédérale', 'Powell', 'Russie', 'Ukraine', 'WeWork', 'GameStop', 'Silicon Valley',
             'Credit Suisse', 'JPMorgan', 'First Republic', 'Nvidia', 'ChatGPT', 'Nasdaq', 'S&P', 'WTI',
             'États-Unis', 'Chine', 'yuan', 'britannique', 'Trésor', 'Inflation Reduction', 'CHIPS',
             'SPAC', 'COVID', 'américain']


def test_anonymized_news_leaks_no_names_or_dates():
    for n in news.NEWS:
        text = f"{n.anon_headline} {n.anon_body}"
        assert not re.search(r'\b(19|20)\d{2}\b', text), text
        for word in FORBIDDEN:
            assert word.lower() not in text.lower(), (word, text)


def test_news_calendar_is_well_formed():
    for n in news.NEWS:
        assert re.fullmatch(r'20(19|20|21|22|23)-(0[1-9]|1[0-2])', n.month)
        assert n.headline and n.body and n.anon_headline and n.anon_body and n.tags
    assert news.news_for('2020-03', anonymize=False)[0]['headline'].startswith('La pandémie')
    assert news.news_for('1999-01') == []


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(fundsim_bp, url_prefix='/api/fundsim')
    return app.test_client()


def test_data_status_lists_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(Config, 'FUNDSIM_DATA_DIR', str(tmp_path))
    body = client.get('/api/fundsim/data/status').get_json()['data']
    assert len(body['missing']) == len(proxies.SERIES) and body['host_to_allow'] == 'fred.stlouisfed.org'


def test_news_endpoint_anonymize_hides_months(client):
    plain = client.get('/api/fundsim/news').get_json()['data']
    anon = client.get('/api/fundsim/news?anonymize=true').get_json()['data']
    assert all(i['month'] for i in plain) and all(i['month'] is None for i in anon)
