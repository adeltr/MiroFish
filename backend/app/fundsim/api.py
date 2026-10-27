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
    start, end, anonymize, social (native|oasis|off), oasis_reactions (rule|llm), oasis_llm_budget, max_calls. Les runs LLM longs deviendront asynchrones à l'étape 7."""
    from flask import request
    from .agents import build_population
    from .data import fred
    from .data.scenario import build_scenario, load_series_returns, synthetic_series_returns
    from .oasis_compat import OasisUnavailable
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
    social = body.get('social', 'native')
    social = 'native' if social is True else 'off' if social is False else social
    if social not in ('native', 'oasis', 'off') or body.get('oasis_reactions', 'rule') not in ('rule', 'llm'):
        return jsonify({'success': False, 'error': "social ∈ {native, oasis, off}, oasis_reactions ∈ {rule, llm}"}), 400
    try:
        result = run_simulation(build_population(seed), scenario, llm,
                                data_source='synthetic' if data == 'synthetic' else 'fred-cache', social=social,
                                oasis_reactions=body.get('oasis_reactions', 'rule'),
                                oasis_llm_budget=int(body.get('oasis_llm_budget', 150)))
    except OasisUnavailable as exc:
        return jsonify({'success': False, 'error': str(exc)}), 503
    from .analysis import allocation_report
    data_out = {'allocations': allocation_report(result)}
    if body.get('detail'):                 # décisions mensuelles, fil social, etc. : seulement à la demande
        data_out['detail'] = result.to_dict()
    return jsonify({'success': True, 'data': data_out})


@fundsim_bp.route('/compare', methods=['POST'])
def compare_scenarios():
    """Compare des variantes sur la même population et les mêmes marchés, en termes d'ALLOCATIONS.
    Corps : seed, data (cache|synthetic), start, end, variants [{name, social, news, anonymize, mode,
    oasis_reactions}], reference {archétype: {strategy, theme, private_share}} (optionnel)."""
    from flask import request
    from .analysis import compare_variants, run_variants
    from .data import fred
    from .data.scenario import load_series_returns, synthetic_series_returns
    from .oasis_compat import OasisUnavailable

    body = request.get_json(silent=True) or {}
    variants = body.get('variants') or []
    names = [v.get('name') for v in variants]
    if not (2 <= len(variants) <= 6) or not all(names) or len(set(names)) != len(names):
        return jsonify({'success': False, 'error': "2 à 6 variantes, chacune avec un nom unique"}), 400
    if any(v.get('social', 'native') not in ('native', 'oasis', 'off') or v.get('mode', 'rule') not in ('rule', 'llm')
           for v in variants):
        return jsonify({'success': False, 'error': "social ∈ {native, oasis, off}, mode ∈ {rule, llm}"}), 400
    if any(v.get('mode') == 'llm' for v in variants) and not Config.LLM_API_KEY and not Config.FUNDSIM_DRY_RUN:
        return jsonify({'success': False, 'error': 'LLM_API_KEY non configurée'}), 400
    seed, data = int(body.get('seed', 7)), body.get('data', 'cache')
    start, end = body.get('start', '2019-01'), body.get('end', '2023-12')
    try:
        series = (synthetic_series_returns(start, end, seed) if data == 'synthetic'
                  else load_series_returns(allow_network=False))
        results = run_variants(seed, series, start, end, variants,
                               'synthetic' if data == 'synthetic' else 'fred-cache',
                               llm_factory=lambda: FundSimLLM(max_calls=int(body.get('max_calls', 500))))
    except fred.DataError as exc:
        return jsonify({'success': False, 'error': str(exc), 'host_to_allow': 'fred.stlouisfed.org'}), 503
    except OasisUnavailable as exc:
        return jsonify({'success': False, 'error': str(exc)}), 503
    return jsonify({'success': True, 'data': compare_variants(results, variants, body.get('reference'))})
