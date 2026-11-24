from membench.evaluate import evaluate, oracle_answer
from membench.memory import (
    BM25,
    AttributeRouter,
    FactStore,
    Hybrid,
    Recency,
    SimulatedExtractor,
    Snippet,
    n_tokens,
)
from membench.synth import Question, make_dataset, make_history


def test_history_ground_truth_consistent():
    h = make_history(0, n_sessions=15, seed=3)
    days = [t.day for t in h.turns]
    assert days == sorted(days)
    stated = {}
    for t in h.turns:
        if t.fact:
            stated.setdefault(t.fact[0], []).append(t.fact[1])
    for q in h.questions:
        if q.qtype in ("current", "updated"):
            assert stated[q.attr][-1] == q.answer
        if q.qtype == "updated":
            assert len(stated[q.attr]) >= 2
        if q.qtype == "previous":
            assert stated[q.attr][-2] == q.answer
        if q.qtype == "absent":
            assert q.attr not in stated and q.answer is None


def test_generation_is_deterministic():
    a, b = make_history(1, seed=7), make_history(1, seed=7)
    assert [t.text for t in a.turns] == [t.text for t in b.turns]


def test_oracle_reader_semantics():
    q = Question("updated", "city", "?", "Berlin", 99, 50)
    ctx = [Snippet("a", 10, ("city", "Pune")), Snippet("b", 50, ("city", "Berlin"))]
    assert oracle_answer(q, ctx) == "Berlin"
    assert oracle_answer(Question("previous", "city", "?", "Pune", 99, 10), ctx) == "Pune"
    distract = [Snippet("c", 60, None, mention=("city", "Osaka"))]
    assert oracle_answer(q, ctx + distract) == "Berlin"
    assert oracle_answer(q, ctx + distract, gullible=True) == "Osaka"
    assert oracle_answer(Question("absent", "pet", "?", None, 99, None), ctx) is None


def test_recency_respects_budget_and_keeps_latest():
    h = make_history(0, seed=0)
    m = Recency()
    m.ingest(h)
    ctx = m.context(h.questions[0], 50)
    assert sum(n_tokens(s.text) for s in ctx) <= 50
    assert ctx[-1].text.endswith(h.turns[-1].text)


def test_bm25_finds_lexical_match():
    h = make_history(0, seed=0)
    m = BM25()
    m.ingest(h)
    stmt = next(t for t in h.turns if t.fact)
    top = m.ranked(stmt.text)[0]
    assert m.turns[top].text == stmt.text


def test_router_top2_accuracy_high():
    r = AttributeRouter()
    qs = [q for h in make_dataset(10, seed=1) for q in h.questions]
    assert sum(q.attr in r.route(q.text, 2) for q in qs) / len(qs) > 0.95


def test_fact_store_supersession_and_noise():
    data = make_dataset(10, seed=0)
    perfect = evaluate(FactStore, data, 256)
    assert perfect.overall > 0.97
    noisy = evaluate(lambda: FactStore(SimulatedExtractor(0.0, 0.5, seed=0)), data, 256)
    assert noisy.accuracy["absent"] < perfect.accuracy["absent"]


def test_hybrid_recovers_extractor_misses():
    data = make_dataset(20, seed=2)
    ex = SimulatedExtractor(miss_rate=0.5, seed=0)
    fs = evaluate(lambda: FactStore(ex), data, 512)
    hy = evaluate(lambda: Hybrid(FactStore(ex)), data, 512)
    assert hy.overall > fs.overall
