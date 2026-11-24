"""Memory strategies. Each one ingests a user's history and, given a question and a
token budget, returns the context snippets an LLM would see.

* ``Recency``      — the last N tokens of transcript (what a plain chat loop does).
* ``BM25``         — lexical retrieval over past turns.
* ``Dense``        — LSA embeddings (TF-IDF → SVD) + cosine; a CPU-only stand-in
  for a neural embedder, which keeps the benchmark offline and deterministic.
* ``BM25Recency``  — BM25 candidates re-scored with a recency bonus.
* ``FactStore``    — write-time extraction into ``(attr, value, valid_from, valid_to)``
  records with supersession, plus query routing to attributes.
* ``Hybrid``       — fact-store records first, BM25 turns fill the remaining budget.

Snippets carry the ground-truth ``fact`` of the turn they came from (or the record
that was extracted) so an *oracle reader* can grade what the memory surfaced
independently of any LLM's reading ability.
"""

from __future__ import annotations

import math
import random
import re
import zlib
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .synth import DESCRIPTIONS, History, Question, Turn

_TOK = re.compile(r"[a-z0-9]+")
STOP = set(
    "a an the i my me is am are was were to of in on at for and or do does did what which who "
    "where when how any it this that these days now just our we you your with from".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in STOP]


def n_tokens(text: str) -> int:
    return len(text.split())


@dataclass
class Snippet:
    text: str
    day: int
    fact: tuple[str, str] | None  # what this snippet asserts about the user (for grading)
    source: str = "turn"  # "turn" | "record"
    mention: tuple[str, str] | None = None  # value mentioned about someone else (distractor)


def _turn_snippet(t: Turn) -> Snippet:
    return Snippet(f"[day {t.day}] {t.role}: {t.text}", t.day, t.fact, mention=t.mentions)


def _fill(snips: list[Snippet], budget: int) -> list[Snippet]:
    out, used = [], 0
    for s in snips:
        c = n_tokens(s.text)
        if used + c > budget:
            continue
        out.append(s)
        used += c
    return out


class Memory:
    name = "base"

    def ingest(self, history: History) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def context(self, q: Question, budget: int) -> list[Snippet]:  # pragma: no cover
        raise NotImplementedError


class Recency(Memory):
    name = "recency"

    def ingest(self, history: History) -> None:
        self.turns = history.turns

    def context(self, q: Question, budget: int) -> list[Snippet]:
        out, used = [], 0
        for t in reversed(self.turns):
            s = _turn_snippet(t)
            c = n_tokens(s.text)
            if used + c > budget:
                break
            out.append(s)
            used += c
        return out[::-1]


class BM25(Memory):
    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75, user_only: bool = True):
        self.k1, self.b, self.user_only = k1, b, user_only

    def ingest(self, history: History) -> None:
        self.turns = [t for t in history.turns if t.role == "user" or not self.user_only]
        self.docs = [Counter(tokenize(t.text)) for t in self.turns]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = self.lens.mean() if len(self.lens) else 1.0
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        s = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
            for w in q:
                f = d.get(w, 0)
                if f:
                    s[i] += self.idf[w] * f * (self.k1 + 1) / (f + norm)
        return s

    def ranked(self, query: str) -> list[int]:
        s = self.scores(query)
        return [i for i in np.argsort(-s, kind="stable") if s[i] > 0]

    def context(self, q: Question, budget: int) -> list[Snippet]:
        picked = _fill([_turn_snippet(self.turns[i]) for i in self.ranked(q.text)], budget)
        return sorted(picked, key=lambda s: s.day)


class BM25Recency(BM25):
    """BM25 top candidates, re-scored: normalised BM25 + ``beta`` x recency in [0, 1]."""

    name = "bm25+recency"

    def __init__(self, beta: float = 0.5, candidates: int = 30, **kw):
        super().__init__(**kw)
        self.beta, self.candidates = beta, candidates

    def context(self, q: Question, budget: int) -> list[Snippet]:
        s = self.scores(q.text)
        cand = [i for i in np.argsort(-s, kind="stable")[: self.candidates] if s[i] > 0]
        if not cand:
            return []
        smax = s[cand[0]]
        dmax = max(t.day for t in self.turns) or 1
        cand.sort(key=lambda i: -(s[i] / smax + self.beta * self.turns[i].day / dmax))
        picked = _fill([_turn_snippet(self.turns[i]) for i in cand], budget)
        return sorted(picked, key=lambda x: x.day)


class LSAEmbedder:
    """TF-IDF (word 1–2 grams + char 3–5 grams) → truncated SVD → L2-normalised vectors."""

    def __init__(self, dim: int = 128, seed: int = 0):
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.svd = TruncatedSVD(dim, random_state=seed)

    def fit(self, corpus: list[str]) -> LSAEmbedder:
        x = hstack([self.word.fit_transform(corpus), self.char.fit_transform(corpus)])
        self.svd.fit(x)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        x = hstack([self.word.transform(texts), self.char.transform(texts)])
        v = self.svd.transform(x)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


class Dense(Memory):
    name = "dense"

    def __init__(self, embedder: LSAEmbedder):
        self.emb = embedder

    def ingest(self, history: History) -> None:
        self.turns = [t for t in history.turns if t.role == "user"]
        self.vecs = self.emb.encode([t.text for t in self.turns])

    def context(self, q: Question, budget: int) -> list[Snippet]:
        sim = self.vecs @ self.emb.encode([q.text])[0]
        order = np.argsort(-sim, kind="stable")
        picked = _fill([_turn_snippet(self.turns[i]) for i in order], budget)
        return sorted(picked, key=lambda s: s.day)


# --------------------------------------------------------------------------- fact store


class SimulatedExtractor:
    """Stand-in for an LLM fact extractor with *controllable* error rates.

    It reads the generator's ground truth and corrupts it: each true statement is
    missed with probability ``miss_rate``; each distractor mention ("my sister moved
    to Berlin") is wrongly recorded as a user fact with probability ``confusion_rate``.
    This lets the benchmark ask: *how good must the extractor be before structured
    memory beats retrieval?* ``LLMExtractor`` in ``llm.py`` is the real thing.
    """

    def __init__(self, miss_rate: float = 0.0, confusion_rate: float = 0.0, seed: int = 0):
        self.miss, self.conf, self.seed = miss_rate, confusion_rate, seed

    def __call__(self, t: Turn, user_id: int) -> tuple[str, str] | None:
        rng = random.Random(zlib.crc32(f"{self.seed}|{user_id}|{t.session}|{t.text}".encode()))
        if t.fact is not None:
            return None if rng.random() < self.miss else t.fact
        if t.mentions is not None and rng.random() < self.conf:
            return t.mentions
        return None


class AttributeRouter:
    """Route a question to schema attributes by TF-IDF similarity to hand-written descriptions."""

    def __init__(self, descriptions: dict[str, str] = DESCRIPTIONS):
        self.attrs = list(descriptions)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.m = self.vec.fit_transform(list(descriptions.values()))

    def route(self, question: str, k: int = 2) -> list[str]:
        s = (self.m @ self.vec.transform([question]).T).toarray().ravel()
        return [self.attrs[i] for i in np.argsort(-s, kind="stable")[:k]]


@dataclass
class Record:
    attr: str
    value: str
    valid_from: int
    valid_to: int | None = None


class FactStore(Memory):
    name = "fact_store"

    def __init__(self, extractor=None, router: AttributeRouter | None = None, k_attrs: int = 2):
        self.extract = extractor or SimulatedExtractor()
        self.router = router or AttributeRouter()
        self.k = k_attrs

    def ingest(self, history: History) -> None:
        self.records: dict[str, list[Record]] = {}
        for t in history.turns:
            if t.role != "user":
                continue
            f = self.extract(t, history.user_id)
            if f is None:
                continue
            attr, value = f
            recs = self.records.setdefault(attr, [])
            if recs and recs[-1].value == value:
                continue
            if recs:
                recs[-1].valid_to = t.day  # supersede, keep history
            recs.append(Record(attr, value, t.day))

    def snippets(self, attr: str) -> list[Snippet]:
        out = []
        for r in self.records.get(attr, []):
            span = (
                f"since day {r.valid_from}"
                if r.valid_to is None
                else (f"day {r.valid_from}–{r.valid_to}, superseded")
            )
            out.append(
                Snippet(
                    f"user.{attr} = {r.value} ({span})", r.valid_from, (attr, r.value), "record"
                )
            )
        return out

    def context(self, q: Question, budget: int) -> list[Snippet]:
        snips = [s for a in self.router.route(q.text, self.k) for s in self.snippets(a)]
        return _fill(snips, budget)


class Hybrid(Memory):
    name = "hybrid"

    def __init__(self, store: FactStore, retriever: BM25 | None = None):
        self.store, self.ret = store, retriever or BM25Recency()

    def ingest(self, history: History) -> None:
        self.store.ingest(history)
        self.ret.ingest(history)

    def context(self, q: Question, budget: int) -> list[Snippet]:
        recs = self.store.context(q, budget)
        left = budget - sum(n_tokens(s.text) for s in recs)
        return recs + self.ret.context(q, max(0, left))
