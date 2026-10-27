"""Fil social adossé à OASIS (camel-ai) : les engagements des LP deviennent de vrais posts d'une
plateforme OASIS, les autres LP y réagissent (likes, commentaires) et ces réactions, lues dans la base
SQLite d'OASIS, renforcent ou non le poids des posts dans la vue des pairs.

Deux politiques de réaction :
- « rule » : réactions déterministes (aucun appel LLM), probabilité proportionnelle au `herding` du LP ;
- « llm »  : chaque LP réagissant est un SocialAgent OASIS piloté par LLM (LLMAction), plafonné par
  `llm_action_budget` pour maîtriser le coût.
"""

import asyncio
import contextlib
import io
import os
import random
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ...config import Config
from ..agents import LPAgent
from ..oasis_compat import ensure_oasis
from .social import Feed, Post, SocialGraph

COMMENT_PROBABILITY = 0.3        # part des réactions « rule » qui s'accompagnent d'un commentaire
MAX_LLM_REACTORS_PER_MONTH = 4


@dataclass
class OasisStats:
    mode: str = 'rule'
    posts: int = 0
    likes: int = 0
    comments: int = 0
    llm_actions: int = 0


def _quiet():
    """OASIS affiche des profils et des chemins sur stdout : on les écarte."""
    return contextlib.redirect_stdout(io.StringIO())


class OasisSession:
    """Enveloppe synchrone autour de la boucle asyncio d'OASIS (une boucle dédiée par session)."""

    def __init__(self, agents: Sequence[LPAgent], graph: SocialGraph, db_path: str, use_llm: bool):
        oasis = ensure_oasis()
        self.oasis = oasis
        self.db_path = db_path
        self.loop = asyncio.new_event_loop()
        self.index = {a.agent_id: a.index for a in agents}
        self.agents_by_index = {}
        self.env = None
        self.use_llm = use_llm
        with _quiet():
            self.loop.run_until_complete(self._start(agents, graph))

    def _model(self):
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType, ModelType
        if not self.use_llm:
            return ModelFactory.create(model_platform=ModelPlatformType.OPENAI, model_type=ModelType.STUB)
        if not Config.LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY non configurée : impossible de piloter les agents OASIS par LLM")
        return ModelFactory.create(model_platform=ModelPlatformType.OPENAI, model_type=Config.FUNDSIM_MODEL_FAST,
                                   api_key=Config.LLM_API_KEY, url=Config.LLM_BASE_URL)

    async def _start(self, agents, graph: SocialGraph):
        o = self.oasis
        from oasis.social_platform.channel import Channel
        from oasis.social_platform.platform import Platform
        platform = Platform(db_path=self.db_path, channel=Channel(), recsys_type="reddit", allow_self_rating=True,
                            show_score=True, max_rec_post_len=20, refresh_rec_post_count=5)
        agent_graph = o.AgentGraph()
        model = self._model()
        actions = [o.ActionType.CREATE_POST, o.ActionType.CREATE_COMMENT, o.ActionType.LIKE_POST,
                   o.ActionType.DO_NOTHING]
        for a in agents:
            info = o.UserInfo(
                user_name=a.agent_id.lower(), name=a.name, recsys_type="reddit",
                description=a.policy.persona,
                profile={"other_info": {"user_profile": f"{a.name} ({a.archetype}) : {a.policy.persona}",
                                        "gender": "n/a", "age": 0, "mbti": "ISTJ", "country": "US/CA"}})
            agent = o.SocialAgent(agent_id=a.index, user_info=info, model=model, agent_graph=agent_graph,
                                  available_actions=actions)
            agent_graph.add_agent(agent)
            self.agents_by_index[a.index] = agent
        for src, followed in graph.follows.items():
            for dst in followed:
                agent_graph.add_edge(self.index[src], self.index[dst])
        self.env = o.make(agent_graph=agent_graph, platform=platform, database_path=self.db_path)
        await self.env.reset()
        follows = {self.agents_by_index[self.index[src]]: [
            o.ManualAction(action_type=o.ActionType.FOLLOW, action_args={"followee_id": self.index[dst]})
            for dst in followed] for src, followed in graph.follows.items()}
        await self.env.step(follows)

    def step(self, per_agent: Dict[int, list]) -> None:
        if per_agent:
            with _quiet():
                self.loop.run_until_complete(
                    self.env.step({self.agents_by_index[i]: a for i, a in per_agent.items()}))

    def manual(self, kind: str, **args):
        return self.oasis.ManualAction(action_type=getattr(self.oasis.ActionType, kind), action_args=args)

    def llm_action(self):
        return self.oasis.LLMAction()

    def query(self, sql: str, params: tuple = ()) -> list:
        con = sqlite3.connect(self.db_path)
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()

    def close(self) -> None:
        try:
            if self.env is not None:
                self.loop.run_until_complete(self.env.close())
        finally:
            self.loop.close()


@dataclass
class OasisFeed(Feed):
    """Feed structuré (poids déterministes) + plateforme OASIS (réactions de la communauté)."""

    agents: Sequence[LPAgent] = field(default_factory=list)
    seed: int = 0
    reactions: str = 'rule'                 # 'rule' | 'llm'
    llm_action_budget: int = 150
    db_dir: Optional[str] = None
    stats: OasisStats = field(default_factory=OasisStats)
    session: Optional[OasisSession] = field(default=None, repr=False)
    _owns_dir: bool = False
    _queue: List[Post] = field(default_factory=list, repr=False)
    _keys_by_post: Dict[int, List[Tuple[str, str, int]]] = field(default_factory=dict, repr=False)
    _announced: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.reactions not in ('rule', 'llm'):
            raise ValueError("reactions ∈ {rule, llm}")
        if self.db_dir is None:
            self.db_dir, self._owns_dir = tempfile.mkdtemp(prefix='fundsim-oasis-'), True
        else:
            os.makedirs(self.db_dir, exist_ok=True)
        self.rng = random.Random(f"{self.seed}|oasis")
        self.herding = {a.agent_id: a.policy.herding for a in self.agents}
        self.stats.mode = self.reactions
        self.session = OasisSession(self.agents, self.graph, os.path.join(self.db_dir, 'oasis.db'),
                                    use_llm=self.reactions == 'llm')

    def publish(self, post: Post) -> None:
        super().publish(post)
        self._queue.append(post)

    @staticmethod
    def _content(label: str, posts: List[Post]) -> str:
        top = sorted(posts, key=lambda q: (-q.amount, q.fund_id))[:3]
        listing = ", ".join(f"{q.fund_id} ({q.amount:,.0f} M$)" for q in top)
        return f"[{label}] Nouveaux engagements : {listing}" + (f" et {len(posts) - 3} autres" if len(posts) > 3 else "")

    def end_of_month(self, month_index: int) -> None:
        """Publie un post par LP décideur, déclenche les réactions, relit likes et commentaires."""
        s = self.session
        by_author: Dict[str, List[Post]] = {}
        for p in self._queue:
            by_author.setdefault(p.author, []).append(p)
        if by_author:
            pending = {(s.index[a], self._content(ps[0].label, ps)): (a, ps) for a, ps in sorted(by_author.items())}
            s.step({uid: [s.manual('CREATE_POST', content=content)] for (uid, content) in pending})
            self._announced = []
            for post_id, user_id, content in s.query("select post_id, user_id, content from post"):
                hit = pending.get((user_id, content))
                if hit is not None:
                    author, ps = hit
                    self._keys_by_post[post_id] = [(q.author, q.fund_id, q.month_index) for q in ps]
                    self._announced.append((author, post_id, ps))
            self.stats.posts += len(by_author)
            self._react()
        self._queue = []
        self._refresh()

    def _followers_of(self, author: str) -> List[str]:
        return sorted(lp for lp, followed in self.graph.follows.items() if author in followed)

    def _react(self) -> None:
        s = self.session
        plan: Dict[int, list] = {}
        if self.reactions == 'rule':
            for author, post_id, ps in sorted(self._announced, key=lambda t: t[0]):
                top = max(ps, key=lambda q: (q.amount, q.fund_id))
                for follower in self._followers_of(author):
                    if self.rng.random() >= self.herding[follower]:
                        continue
                    acts = plan.setdefault(s.index[follower], [])
                    acts.append(s.manual('LIKE_POST', post_id=post_id))
                    if self.rng.random() < COMMENT_PROBABILITY:
                        acts.append(s.manual('CREATE_COMMENT', post_id=post_id,
                                             content=f"Je suis aussi positionné sur {top.fund_id}."))
        else:
            reactors = sorted({f for a, _, _ in self._announced for f in self._followers_of(a)})
            self.rng.shuffle(reactors)
            for follower in reactors[:MAX_LLM_REACTORS_PER_MONTH]:
                if self.stats.llm_actions >= self.llm_action_budget:
                    break
                plan[s.index[follower]] = [s.llm_action()]
                self.stats.llm_actions += 1
        s.step(plan)

    def _refresh(self) -> None:
        s = self.session
        likes = {pid: n for pid, n in s.query("select post_id, count(*) from like group by post_id")}
        self.endorsements = {key: n for pid, n in likes.items() for key in self._keys_by_post.get(pid, [])}
        self.stats.likes = sum(likes.values())
        comments: Dict[tuple, List[str]] = {}
        top_key = {pid: max(keys, key=lambda k: k[1]) for pid, keys in self._keys_by_post.items() if keys}
        for post_id, content in s.query("select post_id, content from comment order by comment_id"):
            if post_id in top_key:
                comments.setdefault(top_key[post_id], []).append(" ".join(str(content).split()))
        self.comments = comments
        self.stats.comments = sum(len(v) for v in comments.values())

    def oasis_export(self) -> dict:
        return {'mode': self.stats.mode, 'posts': self.stats.posts, 'likes': self.stats.likes,
                'comments': self.stats.comments, 'llm_actions': self.stats.llm_actions}

    def close(self) -> None:
        try:
            if self.session is not None:
                self.session.close()
        finally:
            if self._owns_dir and self.db_dir:
                shutil.rmtree(self.db_dir, ignore_errors=True)
