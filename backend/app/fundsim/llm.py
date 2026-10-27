"""
Client LLM de fundsim : budget d'appels, dry-run, réponses JSON validées.

Passe par le SDK OpenAI avec une base_url configurable, ce qui couvre l'API
compatible OpenAI d'Anthropic (https://api.anthropic.com/v1/) comme les autres
fournisseurs. Aucune dépendance à Zep ni à OASIS.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional

from ..config import Config


class BudgetExceeded(RuntimeError):
    """Le nombre maximal d'appels LLM pour ce run est atteint."""


class LLMJSONError(ValueError):
    """La réponse du modèle n'est pas un JSON exploitable."""


def extract_json(text: str) -> Dict[str, Any]:
    """Extrait un objet JSON d'une réponse, avec ou sans bloc ```json."""
    if not text:
        raise LLMJSONError("réponse vide")
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, ValueError) as exc:
        raise LLMJSONError(f"JSON invalide : {exc}") from exc
    if not isinstance(value, dict):
        raise LLMJSONError("un objet JSON est attendu")
    return value


class FundSimLLM:
    """Appels LLM comptabilisés. `transport` permet d'injecter un faux client en test."""

    def __init__(
        self,
        max_calls: Optional[int] = None,
        dry_run: Optional[bool] = None,
        transport: Optional[Callable[[str, List[Dict[str, str]], float, int], str]] = None,
    ):
        self.max_calls = Config.FUNDSIM_MAX_CALLS_PER_RUN if max_calls is None else max_calls
        self.dry_run = Config.FUNDSIM_DRY_RUN if dry_run is None else dry_run
        self.calls = 0
        self._transport = transport
        self._client = None

    @property
    def remaining(self) -> int:
        return max(self.max_calls - self.calls, 0)

    def _default_transport(self, model: str, messages: List[Dict[str, str]],
                           temperature: float, max_tokens: int) -> str:
        if self._client is None:
            from openai import OpenAI
            if not Config.LLM_API_KEY:
                raise RuntimeError("LLM_API_KEY non configurée")
            self._client = OpenAI(api_key=Config.LLM_API_KEY, base_url=Config.LLM_BASE_URL)
        response = self._client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def complete(self, messages: List[Dict[str, str]], model: Optional[str] = None,
                 temperature: float = 0.3, max_tokens: int = 1024) -> str:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"budget de {self.max_calls} appels atteint")
        self.calls += 1
        if self.dry_run:
            return '{"dry_run": true}'
        transport = self._transport or self._default_transport
        return transport(model or Config.FUNDSIM_MODEL_FAST, messages, temperature, max_tokens)

    def complete_json(self, messages: List[Dict[str, str]], model: Optional[str] = None,
                      temperature: float = 0.3, max_tokens: int = 1024,
                      retries: int = 1) -> Dict[str, Any]:
        """Demande un JSON ; en cas de réponse invalide, redemande jusqu'à `retries` fois."""
        last_error: Optional[Exception] = None
        for _ in range(retries + 1):
            text = self.complete(messages, model, temperature, max_tokens)
            try:
                return extract_json(text)
            except LLMJSONError as exc:
                last_error = exc
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "Réponds uniquement avec un objet JSON valide."},
                ]
        raise LLMJSONError(str(last_error))
