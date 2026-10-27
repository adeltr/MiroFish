"""Archétypes de LP nord-américains. Tous les paramètres sont des HYPOTHÈSES de modélisation
inspirées de pratiques connues de chaque type d'investisseur, pas des données d'institutions réelles."""

from typing import Dict

from .policy import LPPolicy

S = ('buyout', 'growth', 'venture', 'secondaries', 'private_credit', 'infrastructure', 'real_estate')


def _t(*w):
    assert len(w) == len(S) and abs(sum(w) - 1.0) < 1e-9, w
    return dict(zip(S, w))


# aum en millions de dollars
ARCHETYPES: Dict[str, dict] = {
    'us_public_pension': dict(label="Pension publique américaine", aum=80_000, regions=['Californie', 'Texas', 'New York', 'Ohio'],
        policy=LPPolicy(0.15, (0.10, 0.20), _t(.45, .10, .05, .05, .15, .10, .10), 0.025, 0.12, 0.12, herding=0.5,
            persona="Fonds de retraite public : horizon très long, gouvernance par comité, sensible à la presse et à la comparaison avec ses pairs, prudent sur la liquidité.")),
    'canadian_pension': dict(label="Pension canadienne à gestion directe", aum=150_000, regions=['Ontario', 'Québec', 'Alberta'],
        policy=LPPolicy(0.30, (0.22, 0.38), _t(.35, .10, .02, .05, .13, .20, .15), 0.035, 0.10, 0.15, herding=0.2,
            persona="Grande caisse de retraite canadienne : équipes internes, co-investissements, indépendante dans ses jugements, forte préférence pour les actifs réels.")),
    'endowment': dict(label="Dotation universitaire", aum=12_000, regions=['Nouvelle-Angleterre', 'Californie', 'Mid-Atlantic'],
        policy=LPPolicy(0.55, (0.45, 0.65), _t(.25, .15, .30, .08, .07, .05, .10), 0.06, 0.08, 0.10, herding=0.6,
            persona="Dotation universitaire : modèle « endowment », forte allocation au capital-risque, réseau d'anciens et de pairs très influent, tolère l'illiquidité.")),
    'foundation': dict(label="Fondation privée", aum=4_000, regions=['Midwest', 'Côte Est'],
        policy=LPPolicy(0.25, (0.15, 0.35), _t(.25, .10, .15, .10, .20, .10, .10), 0.04, 0.15, 0.12, herding=0.5,
            persona="Fondation : doit verser un décaissement annuel, mandat prudent, sensible aux critères d'impact.")),
    'family_office': dict(label="Family office", aum=2_500, regions=['Floride', 'Californie', 'New York'],
        policy=LPPolicy(0.35, (0.20, 0.50), _t(.20, .25, .25, .05, .10, .05, .10), 0.08, 0.15, 0.25, herding=0.35,
            theme_tilts={'tech': 1.3, 'healthcare': 1.2},
            persona="Family office : décisions rapides, portefeuille concentré, convictions thématiques fortes, accès direct aux GP.")),
    'insurer': dict(label="Assureur", aum=90_000, regions=['Connecticut', 'Illinois', 'Ontario'],
        policy=LPPolicy(0.08, (0.05, 0.12), _t(.05, .00, .00, .05, .50, .25, .15), 0.02, 0.25, 0.10, herding=0.3,
            theme_tilts={'real_assets': 1.2},
            persona="Assureur : passif long à adosser, capital réglementaire contraignant, cherche du rendement de crédit et des flux contractuels.")),
    'sovereign_wealth': dict(label="Fonds souverain", aum=200_000, regions=['Moyen-Orient', 'Asie', 'Alaska'],
        policy=LPPolicy(0.25, (0.15, 0.35), _t(.30, .15, .10, .05, .10, .20, .10), 0.03, 0.10, 0.10, herding=0.15,
            theme_tilts={'energy_transition': 1.2},
            persona="Fonds souverain : très gros tickets, horizon multi-générationnel, investit en Amérique du Nord comme partenaire stratégique.")),
    'fund_of_funds': dict(label="Fonds de fonds / secondaire", aum=8_000, regions=['New York', 'Boston'],
        policy=LPPolicy(0.90, (0.80, 0.97), _t(.25, .10, .15, .25, .10, .05, .10), 0.25, 0.03, 0.08, herding=0.6,
            persona="Fonds de fonds : déploie rapidement le capital de ses clients, forte diversification, achète des parts sur le marché secondaire quand les décotes sont attractives.")),
    'corporate_pension': dict(label="Retraite d'entreprise", aum=20_000, regions=['Midwest', 'Texas'],
        policy=LPPolicy(0.07, (0.03, 0.12), _t(.30, .00, .00, .10, .40, .10, .10), 0.01, 0.30, 0.10, herding=0.4,
            persona="Régime à prestations définies d'entreprise : réduction du risque en cours, besoin de liquidité, privé limité au crédit et aux rendements stables.")),
}

DEFAULT_COUNTS = {
    'us_public_pension': 3, 'canadian_pension': 2, 'endowment': 2, 'foundation': 1, 'family_office': 3,
    'insurer': 2, 'sovereign_wealth': 1, 'fund_of_funds': 2, 'corporate_pension': 1,
}
