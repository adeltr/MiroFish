"""
fundsim : simulation de l'allocation de capital des LP vers les fonds privés.

Module isolé du flux d'origine (Zep + OASIS) pour pouvoir rester synchronisé
avec le dépôt amont.
"""

from flask import Blueprint

fundsim_bp = Blueprint('fundsim', __name__)

from . import api  # noqa: E402, F401
