"""Routes /api/fundsim/* ."""

from flask import jsonify

from . import fundsim_bp
from ..config import Config
from .llm import FundSimLLM


@fundsim_bp.route('/health', methods=['GET'])
def health():
    """État de la configuration, sans jamais exposer de clé."""
    return jsonify({
        'success': True,
        'data': {
            'llm_configured': bool(Config.LLM_API_KEY),
            'llm_base_url': Config.LLM_BASE_URL,
            'model_fast': Config.FUNDSIM_MODEL_FAST,
            'model_strong': Config.FUNDSIM_MODEL_STRONG,
            'max_calls_per_run': Config.FUNDSIM_MAX_CALLS_PER_RUN,
            'dry_run': Config.FUNDSIM_DRY_RUN,
            'zep_required': False,
        },
    })


@fundsim_bp.route('/llm-ping', methods=['POST'])
def llm_ping():
    """Un seul appel LLM pour vérifier la chaîne complète (clé, URL, modèle)."""
    llm = FundSimLLM(max_calls=1)
    try:
        reply = llm.complete_json([
            {"role": "user", "content": 'Réponds uniquement avec {"ok": true}.'},
        ])
    except Exception as exc:  # erreur réseau, clé invalide, JSON invalide…
        return jsonify({'success': False, 'error': f'{type(exc).__name__}: {exc}'}), 502
    return jsonify({'success': True, 'data': {'reply': reply, 'calls': llm.calls}})


@fundsim_bp.route('/data/status', methods=['GET'])
def data_status():
    """Séries FRED déjà en cache et séries encore à télécharger."""
    from .data import fred, proxies
    cached = set(fred.cached_series())
    needed = sorted(proxies.SERIES.values())
    return jsonify({'success': True, 'data': {
        'cached': sorted(cached & set(needed)),
        'missing': [s for s in needed if s not in cached],
        'host_to_allow': 'fred.stlouisfed.org',
    }})


@fundsim_bp.route('/news', methods=['GET'])
def news_calendar():
    """Calendrier de news. ?anonymize=true masque noms propres et dates."""
    from flask import request
    from .data import news
    anon = request.args.get('anonymize', 'false').lower() == 'true'
    items = []
    for n in news.NEWS:
        for item in news.news_for(n.month, anon):
            if item not in [i['news'] for i in items]:
                items.append({'month': None if anon else n.month, 'news': item})
    return jsonify({'success': True, 'data': items})


@fundsim_bp.route('/population', methods=['GET'])
def population():
    """Résumé de la population d'agents (LP et GP) pour une graine donnée."""
    from flask import request
    from .agents import build_population
    seed = request.args.get('seed', default=7, type=int)
    return jsonify({'success': True, 'data': build_population(seed).summary()})


@fundsim_bp.route('/run', methods=['POST'])
def run():
    """Lance une simulation synchrone. Corps JSON : seed, mode (rule|llm), data (cache|synthetic),
    start, end, anonymize, social (défaut true), max_calls. Les runs LLM longs deviendront asynchrones à l'étape 7."""
    from flask import request
    from .agents import build_population
    from .data import fred
    from .data.scenario import build_scenario, load_series_returns, synthetic_series_returns
    from .sim.runner import run_simulation

    body = request.get_json(silent=True) or {}
    seed = int(body.get('seed', 7))
    mode = body.get('mode', 'rule')
    data = body.get('data', 'cache')
    start, end = body.get('start', '2019-01'), body.get('end', '2023-12')
    if mode not in ('rule', 'llm') or data not in ('cache', 'synthetic'):
        return jsonify({'success': False, 'error': "mode ∈ {rule, llm}, data ∈ {cache, synthetic}"}), 400
    if mode == 'llm' and not Config.LLM_API_KEY and not Config.FUNDSIM_DRY_RUN:
        return jsonify({'success': False, 'error': 'LLM_API_KEY non configurée'}), 400
    try:
        series = (synthetic_series_returns(start, end, seed) if data == 'synthetic'
                  else load_series_returns(allow_network=False))
        scenario = build_scenario(series, start, end, bool(body.get('anonymize', False)))
    except fred.DataError as exc:
        return jsonify({'success': False, 'error': str(exc), 'host_to_allow': 'fred.stlouisfed.org'}), 503
    llm = FundSimLLM(max_calls=int(body['max_calls'])) if mode == 'llm' and 'max_calls' in body \
        else (FundSimLLM() if mode == 'llm' else None)
    result = run_simulation(build_population(seed), scenario, llm,
                            data_source='synthetic' if data == 'synthetic' else 'fred-cache',
                            social=bool(body.get('social', True)))
    return jsonify({'success': True, 'data': result.to_dict()})
