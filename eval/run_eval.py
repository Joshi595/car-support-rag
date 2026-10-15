"""Evaluate the pipeline on eval/golden.jsonl and write eval/results.md.

1. Retrieval: Hit@1 / Hit@3 / Hit@k / MRR@10 for each mode (dense, bm25, hybrid) on single-turn questions.
2. End-to-end (full LangGraph, hybrid): answer contains the expected facts, cites a relevant
   chunk, refuses out-of-scope questions; plus whether condensing fixes follow-up retrieval.

Run:  python -m eval.run_eval          (CHAT_MODEL / TOP_K env vars pick model and k)
"""

import json
import time
from pathlib import Path

from rag import config
from rag.graph import build_graph, cited_sources, is_refusal, new_state
from rag.retrieval import MODES, load_retriever

HERE = Path(__file__).parent
CASES = [json.loads(line) for line in (HERE / "golden.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
K = config.TOP_K


def is_relevant(doc, case) -> bool:
    return [doc.metadata["source"], doc.metadata["section"]] in case["relevant"]


def first_hit(docs, case) -> int | None:
    """1-based rank of the first relevant chunk, or None."""
    return next((rank for rank, d in enumerate(docs, 1) if is_relevant(d, case)), None)


def retrieval_report(retriever) -> list[str]:
    """Natural-language questions vs. terse keyword-style queries (acronyms, codes, product terms)."""
    single = [c for c in CASES if c["relevant"] and not c.get("history")]
    groups = {"natural": [c for c in single if c.get("kind") != "keyword"],
              "keyword": [c for c in single if c.get("kind") == "keyword"]}
    out = [f"### Retrieval ({len(groups['natural'])} natural questions, {len(groups['keyword'])} keyword queries)", "",
           f"| Mode | Natural Hit@1 | Natural Hit@{K} | Keyword Hit@1 | Keyword Hit@{K} | MRR@10 (all) |",
           "|---|---|---|---|---|---|"]
    for mode in MODES:
        ranks = {g: [first_hit(retriever.search(c["q"], 10, mode), c) for c in cs] for g, cs in groups.items()}
        hit = lambda g, n: sum(r is not None and r <= n for r in ranks[g]) / len(ranks[g])  # noqa: E731
        every = ranks["natural"] + ranks["keyword"]
        mrr = sum(1 / r for r in every if r) / len(every)
        out.append(f"| {mode} | {hit('natural', 1):.0%} | {hit('natural', K):.0%} | "
                   f"{hit('keyword', 1):.0%} | {hit('keyword', K):.0%} | {mrr:.3f} |")
    return out


def e2e_report(retriever) -> list[str]:
    graph = build_graph(retriever)
    groups = {"single-turn": [], "follow-up": [], "out-of-scope": []}
    failures, latencies, fu_raw, fu_condensed = [], [], 0, 0
    for c in CASES:
        t = time.perf_counter()
        s = graph.invoke(new_state(c["q"], [tuple(h) for h in c.get("history", [])], k=K))
        latencies.append(time.perf_counter() - t)
        answer = s["answer"]
        if not c["relevant"]:
            group, ok, cited_ok = "out-of-scope", is_refusal(answer), None
        else:
            group = "follow-up" if c.get("history") else "single-turn"
            ok = not is_refusal(answer) and all(kw.lower() in answer.lower() for kw in c["keywords"])
            cited_ok = any(is_relevant(d, c) for _, d in cited_sources(answer, s["docs"]))
            if group == "follow-up":
                fu_raw += first_hit(retriever.search(c["q"], K), c) == 1
                fu_condensed += first_hit(s["docs"], c) == 1
                c = {**c, "q": f"{c['q']}  →  {s['standalone']}"}
        groups[group].append((ok, cited_ok))
        if not ok or cited_ok is False:
            failures.append(f"- **{c['q']}** (correct={ok}, cited relevant={cited_ok})  \n  {answer[:300]!r}")
        print(f"[{'PASS' if ok else 'FAIL'}] {c['q']}")

    out = [f"### End-to-end (`{config.CHAT_MODEL}`, hybrid, k={K})", "",
           "| Question type | n | Correct answer | Cites a relevant chunk |", "|---|---|---|---|"]
    for name, rows in groups.items():
        cites = [ci for _, ci in rows if ci is not None]
        cite_col = f"{sum(cites) / len(cites):.0%}" if cites else "–"
        out.append(f"| {name} | {len(rows)} | {sum(ok for ok, _ in rows) / len(rows):.0%} | {cite_col} |")
    n_fu = len(groups["follow-up"])
    out += ["", f"Follow-up retrieval Hit@1: raw question {fu_raw}/{n_fu} → condensed question {fu_condensed}/{n_fu}",
            f"Mean latency per question: {sum(latencies) / len(latencies):.1f}s", "",
            "#### Failures", *(failures or ["None."])]
    return out


if __name__ == "__main__":
    retriever = load_retriever()
    report = "\n".join([f"## Evaluation — embeddings `{config.EMBED_MODEL}`", "",
                        *retrieval_report(retriever), "", *e2e_report(retriever), ""])
    (HERE / "results.md").write_text(report, encoding="utf-8")
    print("\n" + report)
