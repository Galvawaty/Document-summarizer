"""Diagnose dataset + training pipeline issues."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import LABELS, LABEL2ID, ID2LABEL, model_cfg
from src.dataset import (
    normalize_sample, create_bio_tags, tokenize_and_align_labels,
    build_training_samples, load_json_dataset, get_tokenizer
)

def diagnose():
    data = load_json_dataset("data/processed/train.json")
    print(f"\n{'='*60}")
    print(f"  DATASET DIAGNOSIS")
    print(f"{'='*60}")
    print(f"  Total raw samples: {len(data)}")

    # 1. Check text lengths
    text_lens = []
    for item in data:
        text = item.get("text", "")
        text_lens.append(len(text))
    
    print(f"\n--- Text Length Stats ---")
    print(f"  Min: {min(text_lens)}")
    print(f"  Max: {max(text_lens)}")
    print(f"  Avg: {sum(text_lens)/len(text_lens):.0f}")
    print(f"  > 1000 chars: {sum(1 for l in text_lens if l > 1000)}")
    print(f"  > 2000 chars: {sum(1 for l in text_lens if l > 2000)}")

    # 2. Check entity overlap issues in normalization
    print(f"\n--- Entity Alignment Check ---")
    entity_found = {label: 0 for label in LABELS}
    entity_total = {label: 0 for label in LABELS}
    overlap_issues = 0
    empty_text = 0
    
    for i, item in enumerate(data):
        norm = normalize_sample(item)
        text = norm["text"]
        
        if not text.strip():
            empty_text += 1
            continue
        
        labels_dict = item.get("labels", {})
        for label in LABELS:
            val = labels_dict.get(label, "")
            if val:
                entity_total[label] += 1
                # Check if found in text
                if str(val) in text:
                    entity_found[label] += 1

        # Check for overlapping entities
        ents = norm["entities"]
        for j, e1 in enumerate(ents):
            for e2 in ents[j+1:]:
                if e1["start"] < e2["end"] and e2["start"] < e1["end"]:
                    overlap_issues += 1

    print(f"  Empty text samples: {empty_text}")
    print(f"  Overlapping entity pairs: {overlap_issues}")
    print(f"\n  Entity match rate:")
    for label in LABELS:
        total = entity_total[label]
        found = entity_found[label]
        rate = (found/total*100) if total > 0 else 0
        print(f"    {label:<20} {found}/{total} ({rate:.0f}%)")

    # 3. Check tokenization + BIO tag distribution
    print(f"\n--- BIO Tag Distribution (after tokenization) ---")
    samples = build_training_samples(data)
    print(f"  Successfully processed: {len(samples)}/{len(data)}")
    
    bio_counts = {}
    truncated = 0
    total_tokens = 0
    total_labeled = 0
    total_o = 0
    
    tokenizer = get_tokenizer()
    
    for samp in samples:
        labels = samp["labels"]
        input_ids = samp["input_ids"]
        
        # Check if text was truncated (lots of non-padding tokens = likely truncated)
        non_pad = sum(1 for x in input_ids if x != tokenizer.pad_token_id)
        if non_pad >= model_cfg.max_length - 2:  # -2 for CLS/SEP
            truncated += 1
        
        for lid in labels:
            if lid == -100:
                continue
            total_tokens += 1
            tag = ID2LABEL.get(lid, "O")
            bio_counts[tag] = bio_counts.get(tag, 0) + 1
            if tag == "O":
                total_o += 1
            else:
                total_labeled += 1
    
    print(f"  Truncated samples (>= {model_cfg.max_length} tokens): {truncated}/{len(samples)} ({truncated/len(samples)*100:.1f}%)")
    print(f"  Total valid tokens: {total_tokens}")
    print(f"  O tokens: {total_o} ({total_o/total_tokens*100:.1f}%)")
    print(f"  Labeled tokens: {total_labeled} ({total_labeled/total_tokens*100:.1f}%)")
    
    print(f"\n  Per-tag counts:")
    for tag, count in sorted(bio_counts.items(), key=lambda x: -x[1]):
        pct = count / total_tokens * 100
        print(f"    {tag:<25} {count:>6} ({pct:.2f}%)")

    # 4. Check class imbalance
    print(f"\n--- Class Imbalance ---")
    entity_labels = {tag: count for tag, count in bio_counts.items() if tag != "O"}
    if entity_labels:
        max_label = max(entity_labels.values())
        min_label = min(entity_labels.values())
        print(f"  Most common entity tag: {max(entity_labels, key=entity_labels.get)} ({max_label})")
        print(f"  Least common entity tag: {min(entity_labels, key=entity_labels.get)} ({min_label})")
        print(f"  Imbalance ratio: {max_label/min_label:.1f}x")
        print(f"  O vs Entity ratio: {total_o/total_labeled:.1f}:1")

    print(f"\n{'='*60}")

if __name__ == "__main__":
    diagnose()
