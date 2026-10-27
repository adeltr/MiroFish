"""Chargement d'OASIS sans PyTorch.

`camel-oasis` importe torch, transformers et sentence-transformers au chargement du module, alors
que le recommandeur « reddit » (score de popularité) que nous utilisons n'en a pas besoin. Si le vrai
PyTorch est installé, il est utilisé tel quel ; sinon `oasis_stubs/` fournit des substituts inertes
placés en FIN de sys.path (le vrai paquet gagne toujours s'il existe).
"""

import logging
import os
import sys

STUBS_DIR = os.path.join(os.path.dirname(__file__), 'oasis_stubs')
INSTALL_HINT = ("OASIS n'est pas installé. Installation légère (sans PyTorch) : "
                "`uv pip install --no-deps -r backend/requirements-oasis-lite.txt` (Python 3.11 ou 3.12).")


class OasisUnavailable(RuntimeError):
    pass


def ensure_oasis():
    """Retourne le module `oasis` ou lève OasisUnavailable avec la marche à suivre."""
    if STUBS_DIR not in sys.path:
        sys.path.append(STUBS_DIR)          # en fin de chemin : un vrai torch installé reste prioritaire
    try:
        import oasis
    except ImportError as exc:
        raise OasisUnavailable(f"{INSTALL_HINT} ({exc})") from exc
    for name in ('social.agent', 'social.twitter', 'social.rec', 'oasis.env', 'oasis.agent'):
        logging.getLogger(name).setLevel(logging.WARNING)    # OASIS journalise chaque action en INFO
    return oasis


def oasis_available() -> bool:
    try:
        ensure_oasis()
        return True
    except OasisUnavailable:
        return False
