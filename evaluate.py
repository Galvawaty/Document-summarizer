"""
ocr_extractor.py
Ekstraksi teks dari gambar menggunakan OCR (Tesseract)
"""

import cv2
import pytesseract
from PIL import Image
from typing import List
import numpy as np


class OCRExtractor:
    """Ekstraksi teks dari gambar menggunakan OCR"""
    
    def __init__(self, tesseract_cmd: str = None):
        """
        Initialize OCR extractor
        
        Args:
            tesseract_cmd: Path ke tesseract executable (optional)
                          Contoh Windows: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.tesseract_config = '--oem 3 --psm 6'
    
    def extract_text(self, image_path: str, languages: str = 'eng') -> List[str]:
        """
        Ekstraksi teks dari gambar
        
        Args:
            image_path: Path ke file gambar
            languages: Bahasa untuk OCR (default: 'eng')
                      Multiple languages: 'eng+ind' untuk English + Indonesian
        
        Returns:
            List teks yang terdeteksi
        """
        try:
            # Load image
            img = Image.open(image_path)
            img_cv = cv2.imread(image_path)
            
            if img_cv is None:
                print(f"  Warning: Could not load image {image_path}")
                return []
            
            # Preprocessing untuk meningkatkan akurasi OCR
            texts = []
            
            # Method 1: Grayscale + Adaptive Threshold
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            text1 = pytesseract.image_to_string(
                processed, 
                config=self.tesseract_config,
                lang=languages
            )
            texts.append(text1)
            
            # Method 2: Original grayscale
            text2 = pytesseract.image_to_string(
                gray,
                config=self.tesseract_config,
                lang=languages
            )
            texts.append(text2)
            
            # Method 3: Denoising
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            text3 = pytesseract.image_to_string(
                denoised,
                config=self.tesseract_config,
                lang=languages
            )
            texts.append(text3)
            
            # Combine dan parse semua hasil
            all_text = '\n'.join(texts)
            detected_texts = self._parse_ocr_result(all_text)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_texts = []
            for text in detected_texts:
                text_lower = text.lower()
                if text_lower not in seen:
                    seen.add(text_lower)
                    unique_texts.append(text)
            
            return unique_texts
            
        except Exception as e:
            print(f"  Error saat ekstraksi OCR dari {image_path}: {e}")
            return []
    
    def _parse_ocr_result(self, raw_text: str) -> List[str]:
        """Parse dan bersihkan hasil OCR"""
        # Split by lines
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        
        # Filter meaningful texts
        meaningful_texts = []
        for line in lines:
            # Skip jika terlalu pendek
            if len(line) < 2:
                continue
            
            # Skip jika hanya karakter special
            if not any(c.isalnum() for c in line):
                continue
            
            # Skip jika terlalu banyak karakter aneh (kemungkinan noise)
            alnum_ratio = sum(c.isalnum() or c.isspace() for c in line) / len(line)
            if alnum_ratio < 0.5:
                continue
            
            meaningful_texts.append(line)
        
        return meaningful_texts
    
    def extract_text_with_confidence(self, image_path: str, 
                                     languages: str = 'eng',
                                     min_confidence: float = 30.0) -> List[tuple]:
        """
        Ekstraksi teks dengan confidence score
        
        Args:
            image_path: Path ke file gambar
            languages: Bahasa untuk OCR
            min_confidence: Minimum confidence threshold (0-100)
        
        Returns:
            List of (text, confidence) tuples
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                gray,
                config=self.tesseract_config,
                lang=languages,
                output_type=pytesseract.Output.DICT
            )
            
            # Filter by confidence
            results = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = float(data['conf'][i])
                
                if text and conf >= min_confidence:
                    results.append((text, conf))
            
            return results
            
        except Exception as e:
            print(f"  Error saat ekstraksi OCR dengan confidence: {e}")
            return []