import json
from pathlib import Path
from collections import defaultdict, Counter

coco = json.load(open('data/raw/result.json'))
images = coco.get('images', [])
annotations = coco.get('annotations', [])
categories = {c['id']: c['name'] for c in coco.get('categories', [])}

ann_by_img = defaultdict(list)
for a in annotations:
    ann_by_img[a['image_id']].append(a)

dataset = json.load(open('data/raw/dataset.json'))
converted_ids = set(s['meta']['task_id'] for s in dataset)

not_converted = [img for img in images if img['id'] not in converted_ids]
print(f'Total images: {len(images)}')
print(f'Berhasil convert: {len(converted_ids)}')
print(f'Tidak terkonversi: {len(not_converted)}')

# Cek distribusi jumlah anotasi
print('\n=== Distribusi anotasi per gambar (SEMUA) ===')
all_counts = [len(ann_by_img[img['id']]) for img in images]
for k, v in sorted(Counter(all_counts).items()):
    print(f'  {k} anotasi: {v} gambar')

print('\n=== Distribusi anotasi per gambar (TIDAK CONVERT) ===')
nc_counts = [len(ann_by_img[img['id']]) for img in not_converted]
for k, v in sorted(Counter(nc_counts).items()):
    print(f'  {k} anotasi: {v} gambar')

# Apakah gambar yg tidak convert semua punya 0 anotasi valid (labels kosong setelah OCR)?
print('\n=== Contoh 5 gambar TIDAK terkonversi + annotasinya ===')
for img in not_converted[:5]:
    anns = ann_by_img[img['id']]
    cats = [categories.get(a['category_id'], '?') for a in anns]
    print(f"  {Path(img['file_name']).name}")
    print(f"    {len(anns)} anotasi: {cats}")
