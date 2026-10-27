# fundsim

Simulation de l'allocation de capital des LP nord-américains vers des fonds privés.
Moteur déterministe (`engine/`), données (`data/`), agents (`agents/`), boucle (`sim/`).

## Couche sociale

`POST /api/fundsim/run` accepte `social` :

| valeur   | contenu |
|----------|---------|
| `native` | fil léger en Python pur, instantané (défaut) |
| `oasis`  | vraie plateforme **OASIS** (camel-ai) : un post par décision de LP, réactions (likes, commentaires) lues dans la base SQLite d'OASIS et réinjectées dans la vue des pairs |
| `off`    | aucune couche sociale (référence pour l'A/B) |

Avec `oasis`, `oasis_reactions` vaut `rule` (réactions déterministes, aucun appel LLM, ~45 s pour 60 mois)
ou `llm` (agents OASIS pilotés par LLM, plafonnés par `oasis_llm_budget`).

### Installer OASIS sans PyTorch

`camel-oasis` impose torch/CUDA (plusieurs Go) alors que le recommandeur `reddit` n'en a pas besoin.

```bash
cd backend
uv venv --python 3.12 && uv pip install --no-deps -r requirements-oasis-lite.txt   # ≈ 480 Mo
```

Les modules `torch`, `transformers` et `sentence_transformers` sont remplacés par `oasis_stubs/`
(placés en fin de `sys.path` : un vrai PyTorch installé garde la priorité). Limite : seul le
recommandeur `reddit` est disponible ; `twhin-bert` et les recommandeurs personnalisés sont désactivés.
Les tests OASIS sont ignorés automatiquement si le paquet est absent.
