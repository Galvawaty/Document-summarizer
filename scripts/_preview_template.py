import json

with open("data_test/ground_truth_template.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total sampel: {data['metadata']['total_samples']}")
print()

# Show first 3 samples
for s in data["samples"][:3]:
    print(f"--- Sample #{s['index']} ---")
    print(f"  doc_id    : {s['doc_id']}")
    print(f"  split     : {s['split']}")
    print(f"  image_path: {s['image_path']}")
    print(f"  pdf_match : {s['pdf_match']}")
    print(f"  labels:")
    for k, v in s['labels'].items():
        print(f"    {k}: {str(v)[:80]}")
    print(f"  extracted_text: {s['extracted_text'][:200]}...")
    print(f"  ground_truth_text: {repr(s['ground_truth_text'])}")
    print()

# Show sample 38-40
print("--- Samples #38-40 ---")
for s in data["samples"][-3:]:
    print(f"  #{s['index']} | doc_id={s['doc_id']} | img={s['image_path']} | pdf={s['pdf_match']}")
    print(f"    labels: NOMOR_SURAT={s['labels'].get('NOMOR_SURAT', 'N/A')[:60]}")
    print()

print("Template berhasil dibuat!")
print("Sekarang tinggal isi field 'ground_truth_text' secara manual.")
