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
