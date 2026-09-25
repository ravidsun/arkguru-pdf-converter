"""
Phase 3, step 5: evaluate the RAG system with RAGAS on a golden Q&A set.

Metrics: context_precision, context_recall, faithfulness, answer_relevancy.
Golden set lives in phase3_rag/golden/golden_qa.jsonl (50+ pairs recommended).

    python -m phase3_rag.eval_ragas --config config/config.yaml \
        --golden phase3_rag/golden/golden_qa.jsonl
"""
from __future__ import annotations
import argparse, json, yaml
from phase3_rag.retrieve import HybridRetriever
from phase3_rag.serve import build_prompt, ask_ollama


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--golden", default="phase3_rag/golden/golden_qa.jsonl")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))["phase3"]
    retr = HybridRetriever(cfg)
    model = cfg["serve"]["model_tag"]

    gold = [json.loads(l) for l in open(a.golden) if l.strip()]
    questions, answers, contexts, ground_truth = [], [], [], []
    for row in gold:
        hits = retr.search(row["question"])
        ctxs = [h.text for h in hits]
        ans = ask_ollama(model, build_prompt(row["question"], hits), cfg["serve"]["ctx"])
        questions.append(row["question"]); answers.append(ans)
        contexts.append(ctxs); ground_truth.append(row["answer"])

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (context_precision, context_recall,
                               faithfulness, answer_relevancy)
    ds = Dataset.from_dict({"question": questions, "answer": answers,
                            "contexts": contexts, "ground_truth": ground_truth})
    result = evaluate(ds, metrics=[context_precision, context_recall,
                                   faithfulness, answer_relevancy])
    print(result)


if __name__ == "__main__":
    main()
