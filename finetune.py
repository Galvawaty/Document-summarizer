

import cv2
import numpy as np
from typing import Tuple

from src.model import SceneType


class VisualAnalyzer:
    """Analisis konteks visual dari gambar"""
    
    def __init__(self):
        self.edge_threshold_low = 50
        self.edge_threshold_high = 150
    
    def analyze_scene(self, image_path: str) -> Tuple[SceneType, float]:
        """
        Analisis tipe scene dari gambar
        
        Args:
            image_path: Path ke file gambar
        
        Returns:
            Tuple of (SceneType, confidence_score)
        """
        try:
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"  Warning: Could not load image {image_path}")
                return SceneType.UNKNOWN, 0.0
            
            # Extract features
            features = self._extract_visual_features(img)
            
            # Classify scene
            scene_type, confidence = self._classify_scene(**features)
            
            return scene_type, confidence
            
        except Exception as e:
            print(f"  Error saat analisis visual: {e}")
            return SceneType.UNKNOWN, 0.0
    
    def _extract_visual_features(self, img: np.ndarray) -> dict:
        """Extract fitur visual dari gambar"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Color features (HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        avg_hue = np.mean(hsv[:, :, 0])
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_value = np.mean(hsv[:, :, 2])
        
        # Texture features (using standard deviation)
        texture_std = np.std(gray)
        
        # Brightness distribution
        brightness_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        brightness_hist = brightness_hist.flatten() / brightness_hist.sum()
        
        # Entropy (measure of information/complexity)
        entropy = -np.sum(brightness_hist * np.log2(brightness_hist + 1e-7))
        
        # Spatial frequency (another measure of detail/texture)
        row_diff = np.diff(gray.astype(float), axis=0)
        col_diff = np.diff(gray.astype(float), axis=1)
        spatial_freq = np.sqrt(np.mean(row_diff**2) + np.mean(col_diff**2))
        
        return {
            'edge_density': edge_density,
            'avg_hue': avg_hue,
            'avg_saturation': avg_saturation,
            'avg_value': avg_value,
            'texture_std': texture_std,
            'entropy': entropy,
            'spatial_freq': spatial_freq
        }
    
    def _classify_scene(self, edge_density: float, avg_hue: float,
                       avg_saturation: float, avg_value: float,
                       texture_std: float, entropy: float,
                       spatial_freq: float) -> Tuple[SceneType, float]:
        """
        Klasifikasi scene berdasarkan fitur visual
        
        Returns:
            Tuple of (SceneType, confidence)
        """
        
        # Urban Street: High edge density, complex, varied colors
        urban_score = 0.0
        if edge_density > 0.15:
            urban_score += 0.3
        if avg_saturation > 100:
            urban_score += 0.2
        if entropy > 6.5:
            urban_score += 0.3
        if spatial_freq > 15:
            urban_score += 0.2
        
        # Commercial: Moderate edges, bright, saturated
        commercial_score = 0.0
        if 0.10 < edge_density < 0.18:
            commercial_score += 0.25
        if avg_saturation < 100 and avg_saturation > 60:
            commercial_score += 0.25
        if avg_value > 120:
            commercial_score += 0.25
        if entropy > 6.0:
            commercial_score += 0.25
        
        # Residential: Lower edge density, softer colors
        residential_score = 0.0
        if edge_density < 0.10:
            residential_score += 0.3
        if avg_saturation < 80:
            residential_score += 0.3
        if texture_std < 40:
            residential_score += 0.2
        if 5.0 < entropy < 6.5:
            residential_score += 0.2
        
        # Public Space: Open areas, high saturation (greenery/sky)
        public_score = 0.0
        if edge_density < 0.12:
            public_score += 0.25
        if avg_saturation > 120:
            public_score += 0.3
        if avg_value > 100:
            public_score += 0.25
        if spatial_freq < 12:
            public_score += 0.2
        
        # Mixed: Medium values across the board
        mixed_score = 0.0
        if 0.08 < edge_density < 0.15:
            mixed_score += 0.25
        if 80 < avg_saturation < 120:
            mixed_score += 0.25
        if 5.5 < entropy < 7.0:
            mixed_score += 0.25
        if 10 < spatial_freq < 20:
            mixed_score += 0.25
        
        # Determine best match
        scores = {
            SceneType.URBAN_STREET: urban_score,
            SceneType.COMMERCIAL: commercial_score,
            SceneType.RESIDENTIAL: residential_score,
            SceneType.PUBLIC_SPACE: public_score,
            SceneType.MIXED: mixed_score
        }
        
        best_scene = max(scores.items(), key=lambda x: x[1])
        
        if best_scene[1] < 0.4:
            return SceneType.UNKNOWN, 0.3
        
        # Normalize confidence to 0-1 range
        confidence = min(best_scene[1], 1.0)
        
        return best_scene[0], confidence
    
    def get_dominant_colors(self, image_path: str, k: int = 5) -> list:
        """
        Dapatkan warna dominan dari gambar menggunakan k-means
        
        Args:
            image_path: Path ke file gambar
            k: Jumlah warna dominan yang ingin dideteksi
        
        Returns:
            List of dominant colors in BGR format
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            # Reshape image to be a list of pixels
            pixels = img.reshape((-1, 3))
            pixels = np.float32(pixels)
            
            # K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, 
                                           cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert back to uint8
            centers = np.uint8(centers)
            
            return centers.tolist()
            
        except Exception as e:
            print(f"  Error getting dominant colors: {e}")
            return []