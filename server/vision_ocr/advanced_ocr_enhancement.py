#!/usr/bin/env python3
"""
Advanced OCR Enhancement System
Implements machine learning methods, ensemble approaches, and benchmarking
to improve accuracy, speed, and quality of text extraction from receipts.
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

# Traditional OCR
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: OCR dependencies not available. Install with: pip install pytesseract pillow opencv-python")

# Ollama integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Error: ollama package not found. Install with: pip install ollama")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('advanced_ocr_enhancement.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Structured result from text extraction"""
    text: str
    confidence: float
    method: str
    processing_time: float
    quality_score: float
    metadata: Dict[str, Any]


@dataclass
class BenchmarkMetrics:
    """Benchmark metrics for model comparison"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    processing_time: float
    memory_usage: float
    quality_score: float


class TextExtractor(ABC):
    """Abstract base class for text extractors"""
    
    @abstractmethod
    def extract(self, image_path: Path) -> ExtractionResult:
        """Extract text from image"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get extractor name"""
        pass


class TraditionalOCRExtractor(TextExtractor):
    """Traditional OCR using Tesseract with preprocessing"""
    
    def __init__(self, preprocessing_config: Dict = None):
        self.config = preprocessing_config or {
            'resize_factor': 2.0,
            'denoise': True,
            'sharpen': True,
            'contrast_enhance': True,
            'deskew': True
        }
    
    def get_name(self) -> str:
        return "Traditional_OCR"
    
    def extract(self, image_path: Path) -> ExtractionResult:
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = self._preprocess_image(image_path)
            
            # Extract text with different configurations
            results = []
            configs = [
                {'lang': 'eng', 'config': '--psm 6'},
                {'lang': 'eng', 'config': '--psm 8'},
                {'lang': 'eng', 'config': '--psm 3'},
            ]
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image, **config)
                    if text.strip():
                        results.append(text.strip())
                except Exception as e:
                    logger.warning(f"OCR config failed: {e}")
            
            # Combine results
            combined_text = '\n'.join(results) if results else ""
            
            # Calculate confidence based on text length and quality
            confidence = min(1.0, len(combined_text.strip()) / 100.0)
            quality_score = self._calculate_quality_score(combined_text)
            
            return ExtractionResult(
                text=combined_text,
                confidence=confidence,
                method="Traditional_OCR",
                processing_time=time.time() - start_time,
                quality_score=quality_score,
                metadata={'configs_used': len(configs), 'successful_configs': len(results)}
            )
            
        except Exception as e:
            logger.error(f"Traditional OCR failed: {e}")
            return ExtractionResult(
                text="",
                confidence=0.0,
                method="Traditional_OCR",
                processing_time=time.time() - start_time,
                quality_score=0.0,
                metadata={'error': str(e)}
            )
    
    def _preprocess_image(self, image_path: Path) -> Image.Image:
        """Apply advanced preprocessing to improve OCR accuracy"""
        with Image.open(image_path) as img:
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize for better OCR
            if max(img.size) > 2000:
                ratio = 2000 / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array for OpenCV processing
            img_array = np.array(img)
            
            # Apply preprocessing steps
            if self.config['denoise']:
                img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
            
            if self.config['sharpen']:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                img_array = cv2.filter2D(img_array, -1, kernel)
            
            if self.config['contrast_enhance']:
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            if self.config['deskew']:
                img_array = self._deskew_image(img_array)
            
            return Image.fromarray(img_array)
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Deskew image to improve text alignment"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            coords = np.column_stack(np.where(gray > 0))
            angle = cv2.minAreaRect(coords)[-1]
            
            if angle < -45:
                angle = 90 + angle
            
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
        except Exception as e:
            logger.warning(f"Deskew failed: {e}")
            return image
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate quality score based on text characteristics"""
        if not text.strip():
            return 0.0
        
        # Factors for quality scoring
        length_score = min(1.0, len(text) / 500.0)  # Normalize by expected length
        word_count = len(text.split())
        word_score = min(1.0, word_count / 50.0)  # Normalize by expected word count
        
        # Check for common receipt patterns
        receipt_patterns = ['total', 'subtotal', 'tax', 'change', 'debit', 'credit', 'cash']
        pattern_score = sum(1 for pattern in receipt_patterns if pattern.lower() in text.lower()) / len(receipt_patterns)
        
        # Check for price patterns
        price_pattern = r'\$\d+\.\d{2}'
        import re
        price_matches = len(re.findall(price_pattern, text))
        price_score = min(1.0, price_matches / 10.0)
        
        # Combined score
        quality_score = (length_score * 0.3 + word_score * 0.3 + pattern_score * 0.2 + price_score * 0.2)
        
        return quality_score


class VisionModelExtractor(TextExtractor):
    """Vision model-based text extraction"""
    
    def __init__(self, model: str = "llava:7b", prompt_template: str = None):
        self.model = model
        self.prompt_template = prompt_template or self._get_default_prompt()
    
    def get_name(self) -> str:
        return f"Vision_{self.model}"
    
    def extract(self, image_path: Path) -> ExtractionResult:
        start_time = time.time()
        
        try:
            # Load image data
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Make API call
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': self.prompt_template,
                        'images': [image_data]
                    }
                ],
                options={
                    'temperature': 0.1,
                    'num_predict': 2048,
                    'top_k': 10,
                    'top_p': 0.9
                }
            )
            
            content = response.get('message', {}).get('content', '')
            
            # Calculate confidence and quality
            confidence = self._calculate_confidence(content)
            quality_score = self._calculate_quality_score(content)
            
            return ExtractionResult(
                text=content,
                confidence=confidence,
                method=f"Vision_{self.model}",
                processing_time=time.time() - start_time,
                quality_score=quality_score,
                metadata={'model': self.model, 'response_length': len(content)}
            )
            
        except Exception as e:
            logger.error(f"Vision model extraction failed: {e}")
            return ExtractionResult(
                text="",
                confidence=0.0,
                method=f"Vision_{self.model}",
                processing_time=time.time() - start_time,
                quality_score=0.0,
                metadata={'error': str(e)}
            )
    
    def _get_default_prompt(self) -> str:
        return """Extract all text from this receipt image. Focus on:
1. Store/company name
2. Date and time
3. All items with prices
4. Subtotal, tax, and total amounts
5. Payment method
6. Receipt number

Provide the extracted text in a clear, structured format."""

    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence based on response characteristics"""
        if not text.strip():
            return 0.0
        
        # Factors for confidence scoring
        length_score = min(1.0, len(text) / 1000.0)
        structure_score = 1.0 if any(keyword in text.lower() for keyword in ['total', 'items', 'date']) else 0.5
        
        return (length_score * 0.6 + structure_score * 0.4)
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate quality score for vision model output"""
        if not text.strip():
            return 0.0
        
        # Check for structured information
        quality_indicators = {
            'has_store_name': any(word in text.lower() for word in ['walmart', 'target', 'store', 'market']),
            'has_date': bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)),
            'has_total': bool(re.search(r'\$\d+\.\d{2}', text)),
            'has_items': len(text.split('\n')) > 5,
            'has_prices': len(re.findall(r'\$\d+\.\d{2}', text)) > 0
        }
        
        score = sum(quality_indicators.values()) / len(quality_indicators)
        return score


class EnsembleExtractor(TextExtractor):
    """Ensemble approach combining multiple extractors"""
    
    def __init__(self, extractors: List[TextExtractor], weights: List[float] = None):
        self.extractors = extractors
        self.weights = weights or [1.0] * len(extractors)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def get_name(self) -> str:
        return "Ensemble"
    
    def extract(self, image_path: Path) -> ExtractionResult:
        start_time = time.time()
        
        # Run all extractors in parallel
        with ThreadPoolExecutor(max_workers=len(self.extractors)) as executor:
            future_to_extractor = {
                executor.submit(extractor.extract, image_path): extractor 
                for extractor in self.extractors
            }
            
            results = []
            for future in as_completed(future_to_extractor):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Extractor failed: {e}")
        
        if not results:
            return ExtractionResult(
                text="",
                confidence=0.0,
                method="Ensemble",
                processing_time=time.time() - start_time,
                quality_score=0.0,
                metadata={'error': 'All extractors failed'}
            )
        
        # Combine results using weighted voting
        combined_result = self._combine_results(results)
        combined_result.processing_time = time.time() - start_time
        
        return combined_result
    
    def _combine_results(self, results: List[ExtractionResult]) -> ExtractionResult:
        """Combine multiple extraction results using weighted voting"""
        # Weight by confidence and quality
        weighted_texts = []
        total_weight = 0
        
        for i, result in enumerate(results):
            weight = self.weights[i] * result.confidence * result.quality_score
            if weight > 0 and result.text.strip():
                weighted_texts.append((result.text, weight))
                total_weight += weight
        
        if not weighted_texts:
            # Fallback to best single result
            best_result = max(results, key=lambda r: r.confidence * r.quality_score)
            return best_result
        
        # Combine texts based on weights
        combined_text = self._merge_texts([text for text, _ in weighted_texts])
        
        # Calculate combined confidence and quality
        avg_confidence = sum(r.confidence * self.weights[i] for i, r in enumerate(results))
        avg_quality = sum(r.quality_score * self.weights[i] for i, r in enumerate(results))
        
        return ExtractionResult(
            text=combined_text,
            confidence=avg_confidence,
            method="Ensemble",
            processing_time=0,  # Will be set by caller
            quality_score=avg_quality,
            metadata={
                'extractors_used': len(results),
                'weights': self.weights,
                'individual_results': [r.text[:100] + "..." for r in results]
            }
        )
    
    def _merge_texts(self, texts: List[str]) -> str:
        """Merge multiple text extractions intelligently"""
        if not texts:
            return ""
        
        if len(texts) == 1:
            return texts[0]
        
        # Simple approach: use the longest text as base, supplement with unique lines
        base_text = max(texts, key=len)
        base_lines = set(base_text.split('\n'))
        
        merged_lines = list(base_lines)
        
        # Add unique lines from other texts
        for text in texts:
            if text != base_text:
                lines = text.split('\n')
                for line in lines:
                    if line.strip() and line not in base_lines:
                        merged_lines.append(line)
        
        return '\n'.join(merged_lines)


class MLQualityPredictor:
    """Machine learning model to predict extraction quality"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model_path = model_path or "models/quality_predictor.pkl"
        
        if Path(self.model_path).exists():
            self.load_model()
    
    def extract_features(self, image_path: Path, extraction_result: ExtractionResult) -> np.ndarray:
        """Extract features for quality prediction"""
        features = []
        
        # Text-based features
        text = extraction_result.text
        features.extend([
            len(text),
            len(text.split()),
            len(text.split('\n')),
            len([c for c in text if c.isdigit()]),
            len([c for c in text if c.isupper()]),
            len([c for c in text if c == '$']),
            extraction_result.confidence,
            extraction_result.quality_score
        ])
        
        # Image-based features (if available)
        try:
            with Image.open(image_path) as img:
                features.extend([
                    img.size[0], img.size[1],
                    img.size[0] * img.size[1],  # Area
                    img.size[0] / img.size[1] if img.size[1] > 0 else 0,  # Aspect ratio
                ])
        except:
            features.extend([0, 0, 0, 0])
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data: List[Tuple[Path, ExtractionResult, float]]):
        """Train the quality predictor"""
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available for training")
            return
        
        X = []
        y = []
        
        for image_path, result, true_quality in training_data:
            features = self.extract_features(image_path, result)
            X.append(features.flatten())
            y.append(true_quality)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Quality predictor accuracy: {accuracy:.3f}")
        
        # Save model
        self.save_model()
    
    def predict_quality(self, image_path: Path, extraction_result: ExtractionResult) -> float:
        """Predict quality score for extraction result"""
        if self.model is None:
            return extraction_result.quality_score
        
        features = self.extract_features(image_path, extraction_result)
        features_scaled = self.scaler.transform(features)
        
        # Get probability of high quality
        proba = self.model.predict_proba(features_scaled)[0]
        return proba[1] if len(proba) > 1 else proba[0]
    
    def save_model(self):
        """Save the trained model"""
        if self.model is not None:
            Path(self.model_path).parent.mkdir(exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'vectorizer': self.vectorizer
                }, f)
    
    def load_model(self):
        """Load the trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.vectorizer = data['vectorizer']
        except Exception as e:
            logger.warning(f"Could not load model: {e}")


class BenchmarkSuite:
    """Comprehensive benchmarking suite for OCR systems"""
    
    def __init__(self, test_data_dir: str = "Test/benchmark_data"):
        self.test_data_dir = Path(test_data_dir)
        self.results_dir = Path("Results/benchmarks")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define benchmark metrics
        self.metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'processing_time', 'memory_usage', 'quality_score'
        ]
    
    def run_benchmark(self, extractors: List[TextExtractor], 
                     test_images: List[Path] = None) -> Dict:
        """Run comprehensive benchmark on extractors"""
        if test_images is None:
            test_images = list(self.test_data_dir.glob("*.png")) + list(self.test_data_dir.glob("*.jpg"))
        
        if not test_images:
            logger.warning("No test images found for benchmarking")
            return {}
        
        results = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'test_images': [str(img) for img in test_images],
            'extractors': [extractor.get_name() for extractor in extractors],
            'results': {}
        }
        
        for extractor in extractors:
            logger.info(f"Benchmarking {extractor.get_name()}...")
            extractor_results = self._benchmark_extractor(extractor, test_images)
            results['results'][extractor.get_name()] = extractor_results
        
        # Calculate rankings
        results['rankings'] = self._calculate_rankings(results['results'])
        
        # Save results
        self._save_benchmark_results(results)
        
        return results
    
    def _benchmark_extractor(self, extractor: TextExtractor, 
                           test_images: List[Path]) -> Dict:
        """Benchmark a single extractor"""
        results = []
        total_time = 0
        total_memory = 0
        
        for image_path in test_images:
            try:
                # Measure memory usage
                import psutil
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Run extraction
                start_time = time.time()
                result = extractor.extract(image_path)
                processing_time = time.time() - start_time
                
                memory_after = process.memory_info().rss / 1024 / 1024
                memory_usage = memory_after - memory_before
                
                results.append({
                    'image': str(image_path),
                    'text': result.text,
                    'confidence': result.confidence,
                    'quality_score': result.quality_score,
                    'processing_time': processing_time,
                    'memory_usage': memory_usage
                })
                
                total_time += processing_time
                total_memory += memory_usage
                
            except Exception as e:
                logger.error(f"Benchmark failed for {image_path}: {e}")
        
        # Calculate aggregate metrics
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_quality = np.mean([r['quality_score'] for r in results])
        avg_time = total_time / len(results) if results else 0
        avg_memory = total_memory / len(results) if results else 0
        
        return {
            'individual_results': results,
            'aggregate_metrics': {
                'avg_confidence': avg_confidence,
                'avg_quality_score': avg_quality,
                'avg_processing_time': avg_time,
                'avg_memory_usage': avg_memory,
                'total_processing_time': total_time,
                'success_rate': len(results) / len(test_images)
            }
        }
    
    def _calculate_rankings(self, results: Dict) -> Dict:
        """Calculate rankings across all metrics"""
        rankings = {}
        
        for metric in ['avg_confidence', 'avg_quality_score', 'avg_processing_time', 'avg_memory_usage']:
            # Sort by metric (lower is better for time and memory)
            reverse = metric not in ['avg_processing_time', 'avg_memory_usage']
            sorted_extractors = sorted(
                results.items(),
                key=lambda x: x[1]['aggregate_metrics'][metric],
                reverse=reverse
            )
            rankings[metric] = [name for name, _ in sorted_extractors]
        
        return rankings
    
    def _save_benchmark_results(self, results: Dict):
        """Save benchmark results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        output_path = self.results_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Benchmark results saved to: {output_path}")


class AdvancedOCRSystem:
    """Advanced OCR system with ML enhancement and benchmarking"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.extractors = []
        self.quality_predictor = MLQualityPredictor()
        self.benchmark_suite = BenchmarkSuite()
        
        # Initialize extractors
        self._initialize_extractors()
    
    def _initialize_extractors(self):
        """Initialize available extractors"""
        if OCR_AVAILABLE:
            self.extractors.append(TraditionalOCRExtractor())
        
        if OLLAMA_AVAILABLE:
            # Add multiple vision models
            vision_models = ["llava:7b", "granite3.2-vision:latest", "minicpm-v:latest"]
            for model in vision_models:
                try:
                    self.extractors.append(VisionModelExtractor(model))
                except Exception as e:
                    logger.warning(f"Could not initialize {model}: {e}")
        
        # Create ensemble if multiple extractors available
        if len(self.extractors) > 1:
            weights = [1.0] * len(self.extractors)
            self.extractors.append(EnsembleExtractor(self.extractors, weights))
    
    def extract_text(self, image_path: Path, use_ml_enhancement: bool = True) -> ExtractionResult:
        """Extract text with optional ML enhancement"""
        if not self.extractors:
            raise ValueError("No extractors available")
        
        # Run all extractors
        results = []
        for extractor in self.extractors:
            try:
                result = extractor.extract(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Extractor {extractor.get_name()} failed: {e}")
        
        if not results:
            raise ValueError("All extractors failed")
        
        # Select best result
        if use_ml_enhancement and self.quality_predictor.model is not None:
            # Use ML to predict quality and select best result
            for result in results:
                predicted_quality = self.quality_predictor.predict_quality(image_path, result)
                result.quality_score = predicted_quality
        
        # Return best result based on quality score
        best_result = max(results, key=lambda r: r.quality_score * r.confidence)
        
        return best_result
    
    def run_benchmark(self, test_images: List[Path] = None) -> Dict:
        """Run comprehensive benchmark"""
        return self.benchmark_suite.run_benchmark(self.extractors, test_images)
    
    def train_quality_predictor(self, training_data: List[Tuple[Path, ExtractionResult, float]]):
        """Train the quality predictor with labeled data"""
        self.quality_predictor.train(training_data)


def main():
    """Main function for testing the advanced OCR system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced OCR Enhancement System")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--train", action="store_true", help="Train quality predictor")
    
    args = parser.parse_args()
    
    # Initialize system
    system = AdvancedOCRSystem()
    
    if args.benchmark:
        # Run benchmark
        test_images = [Path(args.image)]
        results = system.run_benchmark(test_images)
        print("Benchmark completed. Check Results/benchmarks/ for detailed results.")
    
    elif args.train:
        # Train quality predictor (requires labeled data)
        print("Training quality predictor...")
        # This would require labeled training data
        print("Training requires labeled data. Implement training data collection.")
    
    else:
        # Extract text from single image
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Image {args.image} not found")
            return
        
        try:
            result = system.extract_text(image_path)
            print(f"\nExtraction Results:")
            print(f"Method: {result.method}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Quality Score: {result.quality_score:.3f}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print(f"\nExtracted Text:\n{result.text}")
        except Exception as e:
            print(f"Extraction failed: {e}")


if __name__ == "__main__":
    main()
