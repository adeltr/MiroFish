"""Moteur déterministe de fundsim : flux de fonds privés, portefeuilles LP, contraintes, flux de capital.

Python pur, sans LLM ni réseau : tout est reproductible et testable.
"""

from .cashflows import FundParams, monthly_rate
from .constraints import Limits, ValidationResult, validate_allocation
from .flows import fundraising_shares, trailing_score
from .portfolio import STRATEGIES, Fund, LPAccount, MonthResult, Position

__all__ = [
    'FundParams', 'monthly_rate', 'Limits', 'ValidationResult', 'validate_allocation',
    'fundraising_shares', 'trailing_score', 'STRATEGIES', 'Fund', 'LPAccount',
    'MonthResult', 'Position',
]
