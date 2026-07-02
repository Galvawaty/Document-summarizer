from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score


GT_PATH = Path("data_test/ground_truth.json")
REPORT_PATH = Path("output/ground_truth_eval_report.json")


def load_ground_truth(path: str | Path = GT_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    logger.info(f"Loaded {len(samples)} ground truth samples")
    return samples


def compute_rouge(reference: str, hypothesis: str) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": {
            "precision": scores["rouge1"].precision,
            "recall": scores["rouge1"].recall,
            "fmeasure": scores["rouge1"].fmeasure,
        },
        "rouge2": {
            "precision": scores["rouge2"].precision,
            "recall": scores["rouge2"].recall,
            "fmeasure": scores["rouge2"].fmeasure,
        },
        "rougeL": {
            "precision": scores["rougeL"].precision,
            "recall": scores["rougeL"].recall,
            "fmeasure": scores["rougeL"].fmeasure,
        },
    }


def compute_bertscore_batch(
    references: list[str],
    hypotheses: list[str],
    lang: str = "id",
    verbose: bool = False,
) -> list[dict]:
    P, R, F1 = bert_score(
        hypotheses,
        references,
        lang=lang,
        verbose=verbose,
    )
    results = []
    for p, r, f in zip(P.tolist(), R.tolist(), F1.tolist()):
        results.append({
            "precision": round(p, 6),
            "recall": round(r, 6),
            "f1": round(f, 6),
        })
    return results


def summarize_metrics(per_sample: list[dict]) -> dict:
    rouge_keys = ["rouge1", "rouge2", "rougeL"]
    summary = {}

    for key in rouge_keys:
        vals_p = [s["rouge"][key]["precision"] for s in per_sample]
        vals_r = [s["rouge"][key]["recall"] for s in per_sample]
        vals_f = [s["rouge"][key]["fmeasure"] for s in per_sample]
        summary[key] = {
            "mean_precision": round(np.mean(vals_p), 4),
            "mean_recall": round(np.mean(vals_r), 4),
            "mean_f1": round(np.mean(vals_f), 4),
            "std_f1": round(np.std(vals_f), 4),
            "min_f1": round(min(vals_f), 4),
            "max_f1": round(max(vals_f), 4),
        }

    bs_f1 = [s.get("bertscore", {}).get("f1", 0) for s in per_sample]
    bs_p = [s.get("bertscore", {}).get("precision", 0) for s in per_sample]
    bs_r = [s.get("bertscore", {}).get("recall", 0) for s in per_sample]
    summary["bertscore"] = {
        "mean_f1": round(np.mean(bs_f1), 4),
        "std_f1": round(np.std(bs_f1), 4),
        "min_f1": round(min(bs_f1), 4),
        "max_f1": round(max(bs_f1), 4),
        "mean_precision": round(np.mean(bs_p), 4),
        "mean_recall": round(np.mean(bs_r), 4),
    }

    return summary


def main():
    logger.info("=" * 70)
    logger.info("  GROUND TRUTH EVALUATION")
    logger.info("=" * 70)

    # ── Load ground truth ──────────────────────────────────────
    samples = load_ground_truth()
    total = len(samples)

    # ── Filter samples with ground_truth ────────────────────
    valid = [s for s in samples if s.get("ground_truth", "").strip()]
    logger.info(f"  Samples with ground_truth: {len(valid)}/{total}")

    if not valid:
        logger.error("Tidak ada sampel dengan ground_truth! Evaluasi dibatalkan.")
        sys.exit(1)

    # ── Load model ─────────────────────────────────────────────
    logger.info("[1/3] Loading NER model...")
    from src.inference import load_model, run_ner
    load_model()
    logger.info("  Model loaded.")

    # ── Run inference + generate summary ──────────────────────
    logger.info("[2/3] Running inference on {} samples...".format(len(valid)))
    from src.postprocess import build_output_json

    system_summaries = []
    expert_summaries = []
    per_sample_details = []

    for i, s in enumerate(valid, 1):
        idx = s["index"]
        text = s.get("extracted_text", "")
        expert = s["ground_truth"].strip()

        logger.info(f"  [{i}/{len(valid)}] Sample #{idx}...")

        try:
            entities = run_ner(text)
        except Exception as e:
            logger.error(f"    ✗ NER failed: {e}")
            entities = {}

        try:
            output = build_output_json(
                raw_entities=entities,
                raw_text=text,
                pdf_path=s.get("pdf_match", "") or "",
            )
            system_summary = output.get("paragraph_summary", "").strip()
        except Exception as e:
            logger.error(f"    ✗ build_output_json failed: {e}")
            system_summary = ""

        if not system_summary:
            logger.warning(f"    ⚠ Empty paragraph_summary for sample #{idx}")

        system_summaries.append(system_summary)
        expert_summaries.append(expert)

        per_sample_details.append({
            "index": idx,
            "doc_id": s.get("doc_id", "N/A"),
            "split": s.get("split", ""),
            "filled_labels": s.get("filled_labels", 0),
            "system_paragraph_summary": system_summary,
            "expert_ground_truth": expert,
        })

    # ── Compute ROUGE ──────────────────────────────────────────
    logger.info("[3/3] Computing ROUGE and BERTScore...")

    rouge_results = []
    for i, (ref, hyp) in enumerate(zip(expert_summaries, system_summaries)):
        if not hyp.strip():
            logger.warning(f"  Sample #{per_sample_details[i]['index']}: empty system summary, ROUGE = 0")
            scores = {
                "rouge1": {"precision": 0, "recall": 0, "fmeasure": 0},
                "rouge2": {"precision": 0, "recall": 0, "fmeasure": 0},
                "rougeL": {"precision": 0, "recall": 0, "fmeasure": 0},
            }
        else:
            scores = compute_rouge(ref, hyp)
        rouge_results.append(scores)
        per_sample_details[i]["rouge"] = scores

    # ── Compute BERTScore ─────────────────────────────────────
    logger.info("  Computing BERTScore (lang=id)...")
    try:
        bs_results = compute_bertscore_batch(
            expert_summaries, system_summaries, verbose=True
        )
        for i, bs in enumerate(bs_results):
            per_sample_details[i]["bertscore"] = bs
    except Exception as e:
        logger.error(f"  BERTScore failed: {e}")
        bs_results = [{"precision": 0, "recall": 0, "f1": 0}] * len(expert_summaries)
        for i in range(len(per_sample_details)):
            per_sample_details[i]["bertscore"] = {
                "precision": 0, "recall": 0, "f1": 0, "error": str(e)
            }

    # ── Summary ───────────────────────────────────────────────
    summary = summarize_metrics(per_sample_details)

    report = {
        "metadata": {
            "total_samples": total,
            "evaluated_samples": len(valid),
            "model_path": "models/checkpoints/indobert-ner-finetuned",
        },
        "aggregate": summary,
        "per_sample": per_sample_details,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"\nReport saved: {REPORT_PATH}")

    # ── Print summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS (System Summary vs Expert Pakar)")
    print("=" * 70)

    print(f"\n  {'Metric':<20} {'F1':>8} {'Prec.':>8} {'Recall':>8}")
    print(f"  {'-'*44}")

    for key in ["rouge1", "rouge2", "rougeL"]:
        s = summary[key]
        print(f"  {key:<20} {s['mean_f1']:>8.4f} {s['mean_precision']:>8.4f} {s['mean_recall']:>8.4f}")

    bs = summary["bertscore"]
    print(f"  {'bertscore':<20} {bs['mean_f1']:>8.4f} {bs['mean_precision']:>8.4f} {bs['mean_recall']:>8.4f}")

    print("\n  Top-3 Best Samples (by BERTScore F1):")
    sorted_by_bs = sorted(per_sample_details, key=lambda x: x.get("bertscore", {}).get("f1", 0), reverse=True)
    for s in sorted_by_bs[:3]:
        bs_f1 = s.get("bertscore", {}).get("f1", 0)
        rl_f1 = s.get("rouge", {}).get("rougeL", {}).get("fmeasure", 0)
        print(f"    #{s['index']}: BERTScore F1={bs_f1:.4f}, ROUGE-L F1={rl_f1:.4f}")

    print("\n  Bottom-3 Samples (by BERTScore F1):")
    for s in sorted_by_bs[-3:]:
        bs_f1 = s.get("bertscore", {}).get("f1", 0)
        rl_f1 = s.get("rouge", {}).get("rougeL", {}).get("fmeasure", 0)
        print(f"    #{s['index']}: BERTScore F1={bs_f1:.4f}, ROUGE-L F1={rl_f1:.4f}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
