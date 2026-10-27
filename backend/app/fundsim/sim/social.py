"""Couche sociale légère : graphe de suivi entre LP, fil de posts, influence, vue des pairs.

Principe proche d'un réseau social simulé (qui suit qui, qui est influent, ce que chacun a vu),
sans dépendance lourde. Déterministe pour une graine donnée.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from ..agents import LPAgent, Population

PRESTIGE = {   # poids d'influence supposé de chaque type d'investisseur auprès de ses pairs
    'endowment': 1.5, 'canadian_pension': 1.4, 'us_public_pension': 1.0, 'sovereign_wealth': 1.0,
    'fund_of_funds': 0.9, 'foundation': 0.7, 'family_office': 0.6, 'insurer': 0.6, 'corporate_pension': 0.5,
}
SAME_ARCHETYPE_BONUS = 3.0     # on suit plus volontiers ses semblables
FOLLOWS_PER_LP = 4
LOOKBACK_MONTHS = 3
FEED_EXPORT_LIMIT = 200
ENDORSE_GAIN = 0.25          # chaque « like » de la communauté augmente de 25 % le poids d'un post
SNIPPET_MAX_CHARS = 200


@dataclass(frozen=True)
class Post:
    month_index: int
    label: str
    author: str
    fund_id: str
    share: float          # part du budget de l'auteur engagée dans ce fonds
    amount: float
    text: str


@dataclass
class PeerView:
    weights: Dict[str, float]     # fonds -> part pondérée par l'influence (somme = 1)
    snippets: List[str]
    n_peers: int

    def summary(self, funds: Mapping, herding: float, top: int = 5) -> str:
        if not self.weights:
            return ""
        ranked = sorted(self.weights.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        lines = [f"- {fid} ({funds[fid].strategy}, {funds[fid].theme}) : {w:.0%} de l'activité récente de tes pairs"
                 for fid, w in ranked if fid in funds]
        lines += [f"- Un pair écrit : « {s} »" for s in self.snippets]
        head = (f"{self.n_peers} pairs que tu suis ont engagé récemment. Ta propension à suivre tes pairs est de "
                f"{herding:.0%} (0 % = tu décides seul, 100 % = tu les imites).")
        return head + "\n" + "\n".join(lines)


@dataclass
class SocialGraph:
    follows: Dict[str, List[str]]
    influence: Dict[str, float]


def build_graph(pop: Population, seed: int, follows_per_lp: int = FOLLOWS_PER_LP) -> SocialGraph:
    rng = random.Random(f"{seed}|graph")
    influence = {a.agent_id: PRESTIGE[a.archetype] * math.log10(max(a.account.total_assets, 10.0))
                 for a in pop.lps}
    follows: Dict[str, List[str]] = {}
    for a in pop.lps:
        pool = {b.agent_id: influence[b.agent_id] * (SAME_ARCHETYPE_BONUS if b.archetype == a.archetype else 1.0)
                for b in pop.lps if b.agent_id != a.agent_id}
        chosen: List[str] = []
        for _ in range(min(follows_per_lp, len(pool))):
            ids = sorted(pool)
            pick = rng.choices(ids, weights=[pool[i] for i in ids])[0]
            chosen.append(pick)
            del pool[pick]
        follows[a.agent_id] = chosen
    return SocialGraph(follows, influence)


@dataclass
class Feed:
    graph: SocialGraph
    posts: List[Post] = field(default_factory=list)
    endorsements: Dict[tuple, int] = field(default_factory=dict)   # (auteur, fonds, mois) -> likes
    comments: Dict[tuple, List[str]] = field(default_factory=dict)  # (auteur, fonds, mois) -> commentaires

    def publish(self, post: Post) -> None:
        self.posts.append(post)

    def peer_view(self, lp_id: str, month_index: int, lookback: int = LOOKBACK_MONTHS) -> PeerView:
        followed = set(self.graph.follows.get(lp_id, []))
        weights: Dict[str, float] = {}
        seen_peers = set()
        visible = [p for p in self.posts
                   if p.author in followed and month_index - lookback < p.month_index <= month_index]
        for p in visible:
            likes = self.endorsements.get((p.author, p.fund_id, p.month_index), 0)
            weights[p.fund_id] = weights.get(p.fund_id, 0.0) + \
                self.graph.influence[p.author] * p.share * (1.0 + ENDORSE_GAIN * likes)
            seen_peers.add(p.author)
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()} if total else {}
        by_influence = sorted(visible, key=lambda p: (-self.graph.influence[p.author], p.fund_id))
        snippets = [p.text[:SNIPPET_MAX_CHARS] for p in by_influence if p.text][:2]
        for p in by_influence:      # commentaires de la communauté sur les posts visibles
            snippets += [c[:SNIPPET_MAX_CHARS] for c in self.comments.get((p.author, p.fund_id, p.month_index), [])[:1]]
        return PeerView(weights, snippets[:4], len(seen_peers))

    def export(self) -> List[dict]:
        return [dict(month=p.label, author=p.author, fund_id=p.fund_id, share=round(p.share, 3),
                     amount=round(p.amount), text=p.text) for p in self.posts[-FEED_EXPORT_LIMIT:]]
