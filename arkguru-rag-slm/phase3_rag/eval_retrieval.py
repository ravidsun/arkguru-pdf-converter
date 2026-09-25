"""Retrieval-only evaluation against the labelled golden set.

Separate from ``eval_ragas`` on purpose. That module scores generation and needs
Ollama plus ``ragas`` and ``datasets``; this one scores only whether the right
chunks come back, so it runs with nothing but numpy, psycopg and a DSN. Changes
to the retrieval path can therefore be measured without a model or a GPU.

    python -m phase3_rag.eval_retrieval --embedder hashing
    python -m phase3_rag.eval_retrieval --embedder hashing --out baseline.json
    python -m phase3_rag.eval_retrieval --embedder hashing --compare baseline.json

Metrics are reported overall and segmented three ways:

  * ``stratum``    -- catches a change that helps lookup but hurts synthesis
  * ``block_type`` -- tracks table retrieval on its own instead of letting it
    vanish into an average, since tables are only ~7% of the corpus
  * ``source_type`` -- a single bucket today (the corpus is 100% pdf), and the
    hook for later answering whether a web crawl helped or hurt

Leg attribution comes from ``dense_rank`` / ``lexical_rank``, which the
``search_chunks`` SQL function already returns, so a regression can be traced to
the dense or the lexical side rather than guessed at.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase3.eval_retrieval")

RECALL_KS = (5, 10, 20, 50)
# search_chunks column order (see common/datastore._HIT_COLS + the rank columns)
_CHUNK_ID, _RRF, _DENSE_RANK, _LEX_RANK = 0, 10, 11, 12


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def recall_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked_ids: Sequence[str], relevant: set[str],
                    k: int = 10) -> float:
    for i, cid in enumerate(ranked_ids[:k]):
        if cid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Binary-gain nDCG: every labelled chunk is equally relevant."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2)
              for i, cid in enumerate(ranked_ids[:k]) if cid in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
class RetrievalEvaluator:
    def __init__(self, store, embedder, k_dense: int, k_lexical: int,
                 k_candidates: int, rewriter=None):
        self.store = store
        self.embedder = embedder
        self.k_dense = k_dense
        self.k_lexical = k_lexical
        self.k_candidates = k_candidates
        self.rewriter = rewriter

    def _search(self, question: str):
        """Ranked rows for one question, fusing variants when a rewriter is set."""
        variants = [question]
        if self.rewriter is not None:
            variants = self.rewriter.rewrite(question)
        runs = []
        for v in variants:
            qvec = self.embedder.encode([v])[0].tolist()
            runs.append(self.store.search_chunks(
                v, qvec, k_dense=self.k_dense, k_lexical=self.k_lexical,
                k_final=self.k_candidates))
        if len(runs) == 1:
            return runs[0]
        from common.rrf import reciprocal_rank_fusion
        return reciprocal_rank_fusion(runs)

    def evaluate_row(self, row: dict) -> dict:
        rows = self._search(row["question"])
        ranked = [r[_CHUNK_ID] for r in rows]
        relevant = set(row["relevant_chunk_ids"])
        hit_positions = [i for i, cid in enumerate(ranked) if cid in relevant]

        # which leg surfaced the relevant chunks?
        dense_found = lexical_found = 0
        for r in rows:
            if r[_CHUNK_ID] in relevant:
                if r[_DENSE_RANK] is not None:
                    dense_found += 1
                if r[_LEX_RANK] is not None:
                    lexical_found += 1

        out = {
            "stratum": row.get("stratum", "unknown"),
            "block_type": row.get("block_type", "text"),
            "source_type": row.get("source_type", "pdf"),
            "candidates": len(ranked),
            "mrr@10": reciprocal_rank(ranked, relevant, 10),
            "ndcg@10": ndcg_at_k(ranked, relevant, 10),
            "first_hit_rank": (hit_positions[0] + 1) if hit_positions else None,
            "dense_leg_hits": dense_found,
            "lexical_leg_hits": lexical_found,
        }
        for k in RECALL_KS:
            out[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        return out


def aggregate(per_row: Sequence[dict]) -> dict:
    metric_keys = [f"recall@{k}" for k in RECALL_KS] + ["mrr@10", "ndcg@10"]

    def summarise(rows: Sequence[dict]) -> dict:
        d = {"n": len(rows)}
        for m in metric_keys:
            d[m] = round(_mean(r[m] for r in rows), 4)
        d["mean_candidates"] = round(_mean(r["candidates"] for r in rows), 1)
        d["dense_leg_hits"] = sum(r["dense_leg_hits"] for r in rows)
        d["lexical_leg_hits"] = sum(r["lexical_leg_hits"] for r in rows)
        d["found_any"] = sum(1 for r in rows if r["first_hit_rank"] is not None)
        return d

    report = {"overall": summarise(per_row)}
    for dim in ("stratum", "block_type", "source_type"):
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in per_row:
            buckets[r[dim]].append(r)
        report[f"by_{dim}"] = {k: summarise(v) for k, v in sorted(buckets.items())}
    return report


def format_report(report: dict) -> str:
    metric_keys = [f"recall@{k}" for k in RECALL_KS] + ["mrr@10", "ndcg@10"]
    lines = []

    def table(title: str, rows: dict) -> None:
        lines.append(f"\n{title}")
        name_w = max([12] + [len(k) for k in rows])
        head = f"{'segment':<{name_w}} {'n':>4} " + " ".join(
            f"{m:>10}" for m in metric_keys) + f" {'found':>6} {'dense':>6} {'lex':>5}"
        lines.append(head)
        lines.append("-" * len(head))
        for name, d in rows.items():
            lines.append(
                f"{name:<{name_w}} {d['n']:>4} "
                + " ".join(f"{d[m]:>10.4f}" for m in metric_keys)
                + f" {d['found_any']:>6} {d['dense_leg_hits']:>6} {d['lexical_leg_hits']:>5}")

    table("OVERALL", {"all": report["overall"]})
    table("BY STRATUM", report["by_stratum"])
    table("BY BLOCK TYPE", report["by_block_type"])
    table("BY SOURCE TYPE", report["by_source_type"])
    return "\n".join(lines)


def format_diff(base: dict, new: dict) -> str:
    metric_keys = ["recall@10", "recall@50", "mrr@10", "ndcg@10"]
    lines = ["\nDELTA vs baseline (new - baseline)"]
    for section in ("overall", "by_stratum", "by_block_type", "by_source_type"):
        if section == "overall":
            pairs = {"all": (base["overall"], new["overall"])}
        else:
            b, n = base.get(section, {}), new.get(section, {})
            pairs = {k: (b[k], n[k]) for k in sorted(set(b) & set(n))}
        if not pairs:
            continue
        lines.append(f"\n  {section}")
        name_w = max([12] + [len(k) for k in pairs])
        lines.append(f"  {'segment':<{name_w}} " + " ".join(f"{m:>12}" for m in metric_keys))
        for name, (bd, nd) in pairs.items():
            cells = []
            for m in metric_keys:
                d = nd[m] - bd[m]
                cells.append(f"{d:>+12.4f}")
            lines.append(f"  {name:<{name_w}} " + " ".join(cells))
    return "\n".join(lines)


def load_golden(path: str) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    missing = [r for r in rows if not r.get("relevant_chunk_ids")]
    if missing:
        raise SystemExit(
            f"{len(missing)} golden rows have no relevant_chunk_ids; "
            f"regenerate with phase3_rag.build_golden")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Retrieval-only evaluation")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--datastore-config", default="config/datastore.yaml")
    ap.add_argument("--golden", default="phase3_rag/golden/golden_qa.jsonl")
    ap.add_argument("--embedder", choices=("hashing", "sentence_transformer"),
                    default="hashing")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--k-dense", type=int, default=None)
    ap.add_argument("--k-lexical", type=int, default=None)
    ap.add_argument("--k-candidates", type=int, default=None)
    ap.add_argument("--rewriter", default=None,
                    help="noop | lexicon (default: config, else noop)")
    ap.add_argument("--out", default=None, help="write the report as JSON")
    ap.add_argument("--compare", default=None, help="diff against this JSON report")
    ap.add_argument("--label", default=None, help="label recorded in the JSON report")
    a = ap.parse_args(argv)

    import yaml
    from common.datastore_config import open_chunk_store
    from phase3_rag.embedder import Embedder

    cfg = yaml.safe_load(open(a.config))["phase3"]
    rc = cfg.get("retrieval", {})
    k_dense = a.k_dense if a.k_dense is not None else rc.get("top_k_vector", 20)
    k_lexical = a.k_lexical if a.k_lexical is not None else rc.get("top_k_bm25", 20)
    k_candidates = (a.k_candidates if a.k_candidates is not None
                    else rc.get("top_k_candidates", max(k_dense, k_lexical)))

    embedder = Embedder(backend=a.embedder, model_name=a.model, dim=a.dim)
    store = open_chunk_store(a.datastore_config, dim=embedder.dim)

    rewriter = None
    rw_name = a.rewriter or rc.get("rewriter", "noop")
    if rw_name and rw_name != "noop":
        from phase3_rag.rewrite import build_rewriter
        rewriter = build_rewriter(rw_name, cfg)

    golden = load_golden(a.golden)
    log.info("evaluating %d golden rows | embedder=%s k_dense=%d k_lexical=%d "
             "k_candidates=%d rewriter=%s", len(golden), a.embedder, k_dense,
             k_lexical, k_candidates, rw_name)

    ev = RetrievalEvaluator(store, embedder, k_dense, k_lexical, k_candidates,
                            rewriter=rewriter)
    per_row = [ev.evaluate_row(r) for r in golden]
    report = aggregate(per_row)
    report["_meta"] = {
        "label": a.label or rw_name,
        "embedder": a.embedder,
        "k_dense": k_dense,
        "k_lexical": k_lexical,
        "k_candidates": k_candidates,
        "rewriter": rw_name,
        "golden": a.golden,
        "rows": len(golden),
    }

    print(format_report(report))
    if a.compare:
        base = json.loads(Path(a.compare).read_text(encoding="utf-8"))
        print(format_diff(base, report))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("wrote %s", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
