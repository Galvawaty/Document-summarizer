import json
import random
from pathlib import Path

with open("data/processed/test.json", encoding="utf-8") as f:
    test = json.load(f)
with open("data/processed/val.json", encoding="utf-8") as f:
    val = json.load(f)

random.seed(42)
val_sampled = random.sample(val, 14)

print("=== Image filenames in test ===")
for item in test:
    print(f"  {item['meta']['image']}")

print()
print("=== Image filenames in val (sampled) ===")
for item in val_sampled:
    print(f"  {item['meta']['image']}")

print()
print("=== PDF files in data_test ===")
for f in sorted(Path("data_test").iterdir()):
    if f.suffix.lower() in (".pdf", ".docx"):
        print(f"  {f.name}")
