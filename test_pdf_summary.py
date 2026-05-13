"""
test_pdf_summary.py
Script untuk test sumarisasi PDF dengan NER.

Penggunaan:
    python test_pdf_summary.py
"""

from pathlib import Path
from src.pdf_handler import extract_text_from_pdf, pages_to_full_text
from src.inference import load_model, run_ner
from pipeline import _create_summary


def test_pdf_summary(pdf_path):
    """Jalankan sumarisasi pada PDF."""
    
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"Error: File tidak ditemukan: {pdf_path}")
        return
    
    print(f"{'='*70}")
    print(f"TEST SUMARISASI PDF")
    print(f"{'='*70}\n")
    
    # Load model
    print("[1/3] Memuat model NER...")
    load_model()
    print("✓ Model siap\n")
    
    # Ekstrak teks dari PDF
    print("[2/3] Ekstrak teks dari PDF...")
    try:
        pages = extract_text_from_pdf(pdf_path)
        full_text = pages_to_full_text(pages)
        print(f"✓ Ekstraksi berhasil: {len(full_text)} karakter\n")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Jalankan NER
    print("[3/3] Jalankan ekstraksi entitas (NER)...\n")
    entities = run_ner(full_text)
    
    # Tampilkan ringkasan
    summary = _create_summary(pdf_path.name, full_text, entities)
    print(summary)
    
    # Tampilkan entitas detail
    print("ENTITAS YANG DITEMUKAN:")
    print("=" * 70)
    entities_found = {k: v for k, v in entities.items() if v is not None}
    
    if entities_found:
        for label, value in entities_found.items():
            if isinstance(value, list):
                print(f"\n{label}: ({len(value)} ditemukan)")
                for i, item in enumerate(value[:3], 1):
                    truncated = item[:65] + "..." if len(item) > 65 else item
                    print(f"  {i}. {truncated}")
                if len(value) > 3:
                    print(f"  ... dan {len(value)-3} lainnya")
            else:
                truncated = value[:65] + "..." if len(value) > 65 else value
                print(f"\n{label}: {truncated}")
    else:
        print("Tidak ada entitas yang ditemukan")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Path PDF
    pdf_file = r"C:\Users\Nathan\Desktop\SKRIPSI\100 - Surat Balasan Tentang Konfirmasi Gelar Akademik_new format.pdf"
    
    # Jalankan test
    test_pdf_summary(pdf_file)
