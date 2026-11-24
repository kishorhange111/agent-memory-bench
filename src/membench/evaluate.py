"""Grading: an oracle reader over the surfaced context, plus evidence recall.

The **oracle reader** answers using only what the memory put in context, trusting
each snippet's asserted fact (so false facts from a noisy extractor *do* hurt):

* ``current`` / ``updated`` → value of the latest snippet asserting the attribute;
* ``previous`` → the value before that one (by day);
* ``absent`` → correct iff nothing in context asserts the attribute.

It isolates *memory quality* from *reader quality*: an LLM reading the same
context can only do worse. The **gullible reader** is the matching lower bound: it
also believes distractor mentions ("my sister moved to Berlin" → user lives in
Berlin). A real LLM reader lands between the two. ``llm.py`` runs the real end-to-end variant.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .memory import Memory, Snippet, n_tokens
from .synth import History, Question

QTYPES = ("current", "updated", "previous", "absent")


def oracle_answer(q: Question, ctx: list[Snippet], gullible: bool = False) -> str | None:
    seq: list[tuple[int, str]] = sorted(
        (s.day, f[1])
        for s in ctx
        for f in (s.fact, s.mention if gullible else None)
        if f is not None and f[0] == q.attr
    )
    vals: list[str] = []
    for _, v in seq:  # collapse repeats: the value *history*
        if not vals or vals[-1] != v:
            vals.append(v)
    if q.qtype == "previous":
        return vals[-2] if len(vals) >= 2 else None
    return vals[-1] if vals else None


def evidence_hit(q: Question, ctx: list[Snippet]) -> bool | None:
    if q.answer is None:
        return None
    return any(s.day == q.evidence_day and s.fact == (q.attr, q.answer) for s in ctx)


@dataclass
class Report:
    strategy: str
    budget: int
    reader: str
    accuracy: dict[str, float]
    evidence_recall: dict[str, float]
    overall: float
    ctx_tokens: float
    n_questions: int

    def row(self) -> dict:
        return self.__dict__


def evaluate(
    make_memory: Callable[[], Memory],
    data: list[History],
    budget: int,
    name: str | None = None,
    gullible: bool = False,
) -> Report:
    acc, rec = defaultdict(list), defaultdict(list)
    toks = []
    mem_name = None
    for h in data:
        mem = make_memory()
        mem_name = mem.name
        mem.ingest(h)
        for q in h.questions:
            ctx = mem.context(q, budget)
            toks.append(sum(n_tokens(s.text) for s in ctx))
            acc[q.qtype].append(oracle_answer(q, ctx, gullible) == q.answer)
            hit = evidence_hit(q, ctx)
            if hit is not None:
                rec[q.qtype].append(hit)
    all_ = [x for v in acc.values() for x in v]
    return Report(
        strategy=name or mem_name or "?",
        budget=budget,
        reader="gullible" if gullible else "oracle",
        accuracy={k: float(np.mean(acc[k])) for k in QTYPES if acc[k]},
        evidence_recall={k: float(np.mean(rec[k])) for k in QTYPES if rec[k]},
        overall=float(np.mean(all_)),
        ctx_tokens=float(np.mean(toks)),
        n_questions=len(all_),
    )


def bootstrap_ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(xs, dtype=float)
    m = rng.choice(a, (n, len(a))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
