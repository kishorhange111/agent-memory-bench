"""Run the benchmark sweeps and write results.

    membench --users 50 --seeds 0 1 2 --out results

Sweeps:
1. **budget** — strategies x context budget (30-session histories), both readers;
2. **length** — strategies x history length at a fixed 256-token budget;
3. **extractor** — fact store / hybrid x extractor miss and confusion rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import QTYPES, evaluate
from .memory import BM25, BM25Recency, Dense, FactStore, Hybrid, LSAEmbedder, Recency
from .memory import SimulatedExtractor as SimX
from .synth import make_dataset


def strategies(emb: LSAEmbedder) -> dict:
    return {
        "recency": Recency,
        "bm25": BM25,
        "bm25+recency": BM25Recency,
        "dense (LSA)": lambda: Dense(emb),
        "fact_store": FactStore,
        "hybrid": lambda: Hybrid(FactStore()),
    }


def _data(users, seed, sessions):
    data = make_dataset(users, seed=seed, n_sessions=sessions)
    emb = LSAEmbedder(64, seed=seed).fit([t.text for h in data for t in h.turns])
    return data, emb


def _agg(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """Mean and seed-std of overall + per-type accuracy grouped by ``keys``."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rs in groups.items():
        d = dict(zip(keys, g, strict=True))
        d["overall"] = float(np.mean([r["overall"] for r in rs]))
        d["overall_std"] = float(np.std([r["overall"] for r in rs]))
        for t in QTYPES:
            d[t] = float(np.mean([r["accuracy"][t] for r in rs]))
        d["ctx_tokens"] = float(np.mean([r["ctx_tokens"] for r in rs]))
        d["n_seeds"] = len(rs)
        out.append(d)
    return out


def md_table(rows: list[dict], keys: tuple[str, ...]) -> str:
    head = list(keys) + ["overall", *QTYPES, "ctx tokens"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [str(r[k]) for k in keys]
        cells.append(f"{100 * r['overall']:.1f} ± {100 * r['overall_std']:.1f}")
        cells += [f"{100 * r[t]:.1f}" for t in QTYPES]
        cells.append(f"{r['ctx_tokens']:.0f}")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def sweep_budget(users, seeds, budgets, sessions=30) -> list[dict]:
    rows = []
    for seed in seeds:
        data, emb = _data(users, seed, sessions)
        for name, mk in strategies(emb).items():
            for b in budgets:
                for gull in (False, True):
                    r = evaluate(mk, data, b, name=name, gullible=gull).row()
                    rows.append({**r, "seed": seed})
                    print(f"[budget] seed={seed} {name} b={b} {r['reader']}: {r['overall']:.3f}")
    return rows


def sweep_length(users, seeds, lengths, budget=256) -> list[dict]:
    rows = []
    for seed in seeds:
        for n in lengths:
            data, emb = _data(users, seed, n)
            for name, mk in strategies(emb).items():
                r = evaluate(mk, data, budget, name=name).row()
                rows.append({**r, "seed": seed, "sessions": n})
                print(f"[length] seed={seed} sessions={n} {name}: {r['overall']:.3f}")
    return rows


def sweep_extractor(users, seeds, misses, confusions, budget=256, sessions=30) -> list[dict]:
    rows = []
    for seed in seeds:
        data, _ = _data(users, seed, sessions)
        for m in misses:
            for c in confusions:
                mks = {
                    "fact_store": lambda m=m, c=c, s=seed: FactStore(SimX(m, c, s)),
                    "hybrid": lambda m=m, c=c, s=seed: Hybrid(FactStore(SimX(m, c, s))),
                }
                for name, mk in mks.items():
                    r = evaluate(mk, data, budget, name=name).row()
                    rows.append({**r, "seed": seed, "miss": m, "confusion": c})
                    print(f"[extractor] seed={seed} miss={m} conf={c} {name}: {r['overall']:.3f}")
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="membench")
    ap.add_argument("--users", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--budgets", type=int, nargs="+", default=[64, 256, 1024, 4096])
    ap.add_argument("--lengths", type=int, nargs="+", default=[10, 30, 100])
    ap.add_argument("--misses", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--confusions", type=float, nargs="+", default=[0.0, 0.05, 0.2])
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    b = sweep_budget(a.users, a.seeds, a.budgets)
    ln = sweep_length(a.users, a.seeds, a.lengths)
    ex = sweep_extractor(a.users, a.seeds, a.misses, a.confusions)
    (a.out / "raw.json").write_text(json.dumps({"budget": b, "length": ln, "extractor": ex}))

    agg_b = _agg(b, ("reader", "strategy", "budget"))
    agg_l = _agg(ln, ("strategy", "sessions"))
    agg_e = _agg(ex, ("strategy", "miss", "confusion"))
    md = [
        "## Budget sweep (30 sessions)",
        md_table(agg_b, ("reader", "strategy", "budget")),
        "## History-length sweep (budget 256)",
        md_table(agg_l, ("strategy", "sessions")),
        "## Extractor-noise sweep (budget 256, 30 sessions)",
        md_table(agg_e, ("strategy", "miss", "confusion")),
    ]
    (a.out / "results.md").write_text("\n\n".join(md) + "\n")
    (a.out / "summary.json").write_text(
        json.dumps({"budget": agg_b, "length": agg_l, "extractor": agg_e}, indent=1)
    )
    try:
        from .plots import plot_all

        plot_all(agg_b, agg_l, agg_e, a.out)
    except ImportError:
        print("matplotlib not installed; skipping plots")


if __name__ == "__main__":
    main()
