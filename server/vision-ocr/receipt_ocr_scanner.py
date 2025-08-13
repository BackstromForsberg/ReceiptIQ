#!/usr/bin/env python3
"""
Optimized Receipt OCR Scanner - Fast Vision Model Based Receipt Analysis
Optimized for resource-constrained systems without GPUs
"""

import sys
import os
import argparse
import json
import time
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import logging

# Ollama integration
try:
    import ollama
except ImportError:
    print("Error: ollama package not found. Install with: pip install ollama")
    sys.exit(1)

# Optional: Traditional OCR for hybrid approach
try:
    import pytesseract
    from PIL import Image
    TRADITIONAL_OCR_AVAILABLE = True
except ImportError:
    TRADITIONAL_OCR_AVAILABLE = False
    print("Warning: pytesseract not available. Install with: pip install pytesseract pillow")

# Optional: OpenCV for PDF cropping
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: opencv-python not available. Install with: pip install opencv-python")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('receipt_ocr_optimized.log')
    ]
)
logger = logging.getLogger(__name__)


class OptimizedReceiptOCRScanner:
    """Optimized Receipt OCR Scanner with multiple speedup strategies"""
    
    def __init__(self, input_dir: str = "Receipts", 
                 output_dir: str = "Reports", 
                 model: str = "llava:7b",  # Use lighter model by default
                 max_image_size: int = 800,  # Max dimension for resizing
                 use_grayscale: bool = True,  # Convert to grayscale for text
                 use_hybrid_ocr: bool = True,  # Use traditional OCR first
                 compression_quality: int = 85):  # JPEG compression quality
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model = model
        self.max_image_size = max_image_size
        self.use_grayscale = use_grayscale
        self.use_hybrid_ocr = use_hybrid_ocr and TRADITIONAL_OCR_AVAILABLE
        self.compression_quality = compression_quality
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
        # Supported image formats
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        self.pdf_formats = {'.pdf'}
        
        # Load template for structured output
        self.template = self._load_template()
        
        # Check model compatibility at startup
        logger.info(f"Checking compatibility for model: {model}")
        if not self._is_vision_model():
            logger.warning(f"Model {model} may not be a vision model")
            if not self._test_model_compatibility():
                logger.warning(f"Model {model} failed vision compatibility test")
                fallback = self._get_fallback_model()
                if fallback != model:
                    logger.info(f"Suggesting fallback model: "
                               f"{fallback}")
        
        # Performance tracking
        self.processing_stats = {
            'total_images': 0,
            'traditional_ocr_success': 0,
            'vision_model_used': 0,
            'total_time': 0,
            'avg_time_per_image': 0
        }
        
        logger.info(f"Initialized OptimizedReceiptOCRScanner with model: {model}")
        logger.info(f"Max image size: {max_image_size}px, Grayscale: {use_grayscale}")
        logger.info(f"Hybrid OCR: {use_hybrid_ocr}")
    
    def _load_template(self) -> Dict:
        """Load the receipt template for structured output"""
        template_path = Path("Template/llama3.2-vision_template.json")
        if template_path.exists():
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load template: {e}")
        
        # Default template
        return {
            "Company Name": "",
            "Receipt Number": "",
            "Date": "",
            "Time": "",
            "Cashier": "",
            "Store Location": "",
            "Items": [],
            "Subtotal": "",
            "Sales Tax": "",
            "Total": "",
            "Payment Method": "",
            "Change": "",
            "Processing Notes": {
                "Image Quality": "",
                "Text Clarity": "",
                "Layout Complexity": "",
                "Extraction Confidence": ""
            }
        }
    
    def _optimize_image(self, image_path: Path) -> bytes:
        """Optimize image for faster processing"""
        try:
            # Check if it's a PDF file
            if image_path.suffix.lower() in self.pdf_formats:
                logger.info(f"Detected PDF file: {image_path.name}")
                return self._load_pdf_data(image_path)
            
            if not TRADITIONAL_OCR_AVAILABLE:
                # Fallback to basic optimization
                return self._basic_image_optimization(image_path)
            
            # Load image with PIL for optimization
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if image is too large
                if max(img.size) > self.max_image_size:
                    ratio = self.max_image_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"Resized image from {img.size} to {new_size}")
                
                # Convert to grayscale for text-heavy images
                if self.use_grayscale:
                    img = img.convert('L').convert('RGB')
                
                # Save with compression
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=self.compression_quality, optimize=True)
                optimized_data = buffered.getvalue()
                
                original_size = image_path.stat().st_size
                optimized_size = len(optimized_data)
                compression_ratio = (1 - optimized_size / original_size) * 100
                
                logger.info(f"Optimized {image_path.name}: {original_size/1024:.1f}KB -> {optimized_size/1024:.1f}KB ({compression_ratio:.1f}% reduction)")
                
                return optimized_data
                
        except Exception as e:
            logger.error(f"Error optimizing {image_path}: {e}")
            # Fallback to basic optimization
            return self._basic_image_optimization(image_path)
    
    def _basic_image_optimization(self, image_path: Path) -> bytes:
        """Basic image optimization without PIL"""
        try:
            with open(image_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error in basic optimization: {e}")
            raise
    
    def _load_pdf_data(self, pdf_path: Path) -> bytes:
        """Load and optimize PDF data using PyMuPDF"""
        try:
            import fitz  # PyMuPDF
            import cv2
            import numpy as np
            
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            
            if len(doc) == 0:
                raise ValueError("PDF has no pages")
            
            # Get first page
            page = doc[0]
            
            # Convert page to image with high resolution
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL format
            img_data = pix.tobytes("png")
            
            # Convert to OpenCV format for cropping
            nparr = np.frombuffer(img_data, np.uint8)
            opencv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Crop receipt content to remove white space
            cropped_img = self._crop_receipt_content(opencv_img)
            
            # Convert back to PIL for further processing
            cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cropped_rgb)
            
            # Apply same optimizations as regular images
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # Resize if too large
            if max(pil_img.size) > self.max_image_size:
                ratio = self.max_image_size / max(pil_img.size)
                new_size = tuple(int(dim * ratio) for dim in pil_img.size)
                pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to grayscale if enabled
            if self.use_grayscale:
                pil_img = pil_img.convert('L').convert('RGB')
            
            # Save with compression
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG", quality=self.compression_quality, optimize=True)
            
            # Clean up immediately
            doc.close()
            del pix
            del opencv_img
            del cropped_img
            del pil_img
            
            logger.info(f"Optimized PDF {pdf_path.name} with PyMuPDF")
            return buffered.getvalue()
            
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path}: {e}")
            raise
    
    def _analyze_pdf_content(self, pdf_path: Path) -> Dict:
        """Analyze PDF content to determine if it's text-based or image-based"""
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if len(pdf_reader.pages) == 0:
                    return {"type": "empty", "pages": 0}
                
                # Check first page for text content
                first_page = pdf_reader.pages[0]
                text_content = first_page.extract_text()
                
                if text_content and len(text_content.strip()) > 50:
                    # PDF contains extractable text
                    return {
                        "type": "text",
                        "pages": len(pdf_reader.pages),
                        "text_length": len(text_content),
                        "sample_text": text_content[:200] + "..." if len(text_content) > 200 else text_content
                    }
                else:
                    # PDF appears to be image-based
                    return {
                        "type": "image",
                        "pages": len(pdf_reader.pages),
                        "text_length": len(text_content) if text_content else 0
                    }
                    
        except Exception as e:
            logger.warning(f"Could not analyze PDF {pdf_path.name}: {e}")
            # Assume image-based if analysis fails
            return {"type": "image", "pages": 1, "error": str(e)}
    
    def _crop_receipt_content(self, image):
        """Crop receipt content using OpenCV - Conservative approach"""
        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV not available - skipping cropping")
            return image
            
        # Convert to grayscale for better edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply more conservative threshold to preserve content
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        # Use morphological operations to connect nearby text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No content found to crop")
            return image
        
        # Find contours with significant area (filter out noise)
        significant_contours = [c for c in contours if cv2.contourArea(c) > 1000]
        
        if not significant_contours:
            logger.warning("No significant content found to crop")
            return image
        
        # Find the largest contour (likely the receipt)
        largest_contour = max(significant_contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add generous padding to preserve content
        padding = 50
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Crop the image
        cropped = image[y:y+h, x:x+w]
        
        logger.info(f"Cropped from {image.shape[1]}x{image.shape[0]} to {cropped.shape[1]}x{cropped.shape[0]}")
        return cropped

    def _extract_text_from_pdf(self, pdf_path: Path) -> Dict:
        """Extract text directly from PDF if it's text-based"""
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {page_num} ---\n{page_text}"
                
                if not text_content.strip():
                    return {"error": "No text content found in PDF"}
                
                # Parse the extracted text
                parsed_data = self._parse_ocr_text(text_content)
                
                return {
                    "raw_text": text_content,
                    "parsed_data": parsed_data,
                    "extraction_method": "pdf_text_extraction",
                    "confidence": "high" if len(text_content.strip()) > 100 else "low"
                }
                
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return {"error": f"PDF text extraction failed: {e}"}
    
    def _extract_with_traditional_ocr(self, image_path: Path) -> Dict:
        """Extract text using traditional OCR (fast)"""
        try:
            if not TRADITIONAL_OCR_AVAILABLE:
                return {"error": "Traditional OCR not available"}
            
            # Load and optimize image
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize for faster OCR
                if max(img.size) > 1200:  # OCR works well with reasonable sizes
                    ratio = 1200 / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Extract text
                text = pytesseract.image_to_string(img)
                
                if not text.strip():
                    return {"error": "No text extracted", "method": "traditional_ocr"}
                
                # Parse the extracted text
                parsed_data = self._parse_ocr_text(text)
                
                return {
                    "raw_text": text,
                    "parsed_data": parsed_data,
                    "extraction_method": "traditional_ocr",
                    "confidence": "high" if len(text.strip()) > 50 else "low"
                }
                
        except Exception as e:
            logger.error(f"Traditional OCR failed: {e}")
            return {"error": f"Traditional OCR failed: {e}", "method": "traditional_ocr"}
    
    def _parse_ocr_text(self, text: str) -> Dict:
        """Parse text extracted by traditional OCR - only extract what's actually there"""
        try:
            import re
            
            # Start with empty dict, don't use template
            parsed = {}
            
            lines = text.split('\n')
            
            # Extract company name (usually at top)
            for line in lines[:5]:  # Check first 5 lines
                if len(line.strip()) > 3 and not re.search(r'\d', line):
                    parsed["Company Name"] = line.strip()
                    break
            
            # Extract total (look for largest number)
            total_pattern = r'\$?\s*(\d+\.\d{2})'
            totals = re.findall(total_pattern, text)
            if totals:
                parsed["Total"] = max(totals, key=float)
            
            # Extract date - look for actual date patterns
            date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(\d{4}[/-]\d{1,2}[/-]\d{1,2})'
            date_match = re.search(date_pattern, text)
            if date_match:
                parsed["Date"] = date_match.group()
            
            # Extract items (lines with product names)
            items = []
            for line in lines:
                # Look for lines with product names (not just prices)
                if (len(line.strip()) > 5 and 
                    not re.search(r'^(TOTAL|SUBTOTAL|TAX|CHANGE|DEBIT|EFT|ACCOUNT|REF|NETWORK|APPR)', line.upper()) and
                    not re.search(r'^\d{1,2}:\d{2}:\d{2}', line) and  # Skip time stamps
                    not re.search(r'^# ITEMS', line.upper()) and
                    line.strip()):
                    
                    # Clean up the line
                    clean_line = line.strip()
                    if len(clean_line) > 3:
                        items.append({
                            "description": clean_line,
                            "quantity": "1",
                            "unit_price": "",
                            "total": "",
                            "category": ""
                        })
            
            if items:
                parsed["Items"] = items[:20]  # Limit items
            
            # Add raw text for reference
            parsed["raw_text"] = text
            
            return parsed
            
        except Exception as e:
            logger.warning(f"Failed to parse OCR text: {e}")
            return {"raw_text": text}
    
    def extract_receipt_data(self, image_path: Path) -> Dict:
        """Extract receipt data using optimized approach with PDF text/image detection"""
        logger.info(f"Processing {image_path.name} with optimized pipeline")
        start_time = time.time()
        
        try:
            # Special handling for PDF files
            if image_path.suffix.lower() in self.pdf_formats:
                logger.info(f"Processing PDF file: {image_path.name}")
                
                # Step 1: Analyze PDF content type
                pdf_analysis = self._analyze_pdf_content(image_path)
                logger.info(f"PDF analysis: {pdf_analysis}")
                
                if pdf_analysis.get("type") == "text":
                    logger.info("PDF contains extractable text - attempting text extraction")
                    
                    # Step 2: Extract text from PDF
                    text_result = self._extract_text_from_pdf(image_path)
                    
                    if "error" not in text_result:
                        logger.info("Text extraction successful - comparing with image-based processing")
                        
                        # Step 3: Also convert to image for comparison
                        image_result = self._process_pdf_as_image(image_path)
                        
                        # Step 4: Compare results and choose best approach
                        comparison = self._compare_text_vs_image_results(text_result, image_result)
                        logger.info(f"Comparison result: {comparison['winner']}")
                        
                        processing_time = time.time() - start_time
                        comparison["processing_time"] = processing_time
                        comparison["pdf_processing"] = True
                        comparison["pdf_analysis"] = pdf_analysis
                        
                        return comparison
                    else:
                        logger.warning("Text extraction failed, falling back to image processing")
                        return self._process_pdf_as_image(image_path)
                
                else:
                    logger.info("PDF is image-based - converting to image for vision processing")
                    return self._process_pdf_as_image(image_path)
            
            # Regular image processing (existing logic)
            # Strategy 1: Try traditional OCR first (fast)
            if self.use_hybrid_ocr:
                logger.info("Attempting traditional OCR first...")
                ocr_result = self._extract_with_traditional_ocr(image_path)
                
                # Check if traditional OCR was successful
                if "error" not in ocr_result and ocr_result.get("confidence") == "high":
                    self.processing_stats['traditional_ocr_success'] += 1
                    processing_time = time.time() - start_time
                    ocr_result["processing_time"] = processing_time
                    logger.info(f"Traditional OCR successful in {processing_time:.2f}s")
                    return ocr_result
            
            # Strategy 2: Use vision model (slower but more accurate)
            logger.info("Using vision model for extraction...")
            self.processing_stats['vision_model_used'] += 1
            
            # Test model compatibility first
            if not self._test_model_compatibility():
                logger.warning(f"Model {self.model} may not support vision input")
                fallback_model = self._get_fallback_model()
                if fallback_model != self.model:
                    logger.info(f"Switching to fallback model: {fallback_model}")
                    original_model = self.model
                    self.model = fallback_model
                    # Test the fallback model
                    if not self._test_model_compatibility():
                        logger.error("Fallback model also failed vision test")
                        self.model = original_model
                        return {"error": "No compatible vision model available"}
            
            # Optimize image before sending to model
            optimized_image = self._optimize_image(image_path)
            
            # Extract with vision model
            result = self._extract_with_vision_model(optimized_image)
            
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            logger.info(f"Vision model processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            return {
                "error": str(e),
                "file": image_path.name,
                "timestamp": datetime.now().isoformat(),
                "processing_time": time.time() - start_time
            }
    
    def _extract_with_vision_model(self, image_data: bytes) -> Dict:
        """Extract data using vision model with proper error handling for different model types"""
        try:
            # Use ollama.chat() for vision models that don't support tools
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': self._get_optimized_prompt(),
                        'images': [image_data]
                    }
                ]
            )
            
            content = response.get('message', {}).get('content', '')
            
            if not content:
                logger.warning(f"No content received from {self.model}")
                return {"error": "No response content from vision model"}
            
            # Parse the response
            parsed_data = self._parse_chat_response(content)
            
            return {
                "raw_response": content,
                "parsed_data": parsed_data,
                "extraction_method": "vision_model",
                "model_used": self.model
            }
            
        except Exception as e:
            logger.error(f"Vision model extraction failed for {self.model}: {e}")
            # Try to provide more specific error information
            error_msg = str(e)
            if "tool" in error_msg.lower():
                logger.info(f"Model {self.model} may not support tools, "
                          f"using chat method")
                return {"error": f"Tool support issue: {error_msg}"}
            elif "image" in error_msg.lower():
                logger.info(f"Model {self.model} may not support image input")
                return {"error": f"Image support issue: {error_msg}"}
            else:
                return {"error": f"Vision model failed: {error_msg}"}
    
    def _test_model_compatibility(self) -> bool:
        """Test if the model supports vision input"""
        try:
            # Try a simple test with a minimal image
            test_image = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9'
            
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': 'Test message',
                        'images': [test_image]
                    }
                ]
            )
            
            return True
            
        except Exception as e:
            logger.warning(f"Model {self.model} may not support vision: {e}")
            return False
    
    def _get_fallback_model(self) -> str:
        """Get a fallback model if the current one doesn't work"""
        # Get list of available models
        try:
            available_models = [model['name'] for model in ollama.list()['models']]
        except Exception as e:
            logger.warning(f"Could not get available models: {e}")
            available_models = []
        
        # Priority list of vision models
        vision_models = ['llava:latest', 'minicpm-v:latest', 'granite3.2-vision:latest']
        
        # First try vision models that are available
        for model in vision_models:
            if model in available_models and model != self.model:
                logger.info(f"Found available fallback vision model: {model}")
                return model
        
        # If no vision models available, try any available model
        for model in available_models:
            if model != self.model:
                logger.info(f"Found available fallback model: {model}")
                return model
        
        return self.model  # Return original if no fallback available
    
    def _get_model_info(self) -> Dict:
        """Get information about the current model"""
        try:
            # Try to get model info from ollama
            model_info = ollama.show(self.model)
            return {
                "name": model_info.get("name", self.model),
                "family": model_info.get("family", "unknown"),
                "parameter_size": model_info.get("parameter_size", "unknown"),
                "modelfile": model_info.get("modelfile", "")
            }
        except Exception as e:
            logger.warning(f"Could not get model info for {self.model}: {e}")
            return {
                "name": self.model,
                "family": "unknown",
                "parameter_size": "unknown",
                "modelfile": ""
            }
    
    def _is_vision_model(self) -> bool:
        """Check if the current model supports vision"""
        model_info = self._get_model_info()
        modelfile = model_info.get("modelfile", "").lower()
        
        # Check for vision-related keywords in modelfile
        vision_keywords = ['vision', 'llava', 'clip', 'image', 'multimodal']
        return any(keyword in modelfile for keyword in vision_keywords)
    
    def _get_optimized_prompt(self) -> str:
        """Get optimized prompt for faster processing with better compatibility"""
        return """Analyze this receipt image and extract the key information.

Please provide the following information if visible:
- Store/Company name
- Date and time
- Items with prices (if listed)
- Subtotal
- Tax amount
- Total amount
- Payment method

If any information is not visible or unclear, please indicate that.

Format your response clearly and be specific about what you can see in the image."""
    
    def _parse_chat_response(self, content: str) -> Dict:
        """Parse structured data from chat response - only extract what's actually mentioned"""
        try:
            import re
            
            # Try to extract JSON
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, content, re.DOTALL)
            
            if json_match:
                try:
                    json_data = json.loads(json_match.group())
                    # Validate that the JSON contains actual data, not placeholders
                    cleaned_json = {}
                    for key, value in json_data.items():
                        if value and str(value).strip() and str(value).lower() not in ['n/a', 'unknown', 'not found', '']:
                            cleaned_json[key] = value
                    if cleaned_json:
                        return cleaned_json
                except json.JSONDecodeError:
                    pass
            
            # Fallback parsing - only extract what's explicitly mentioned
            parsed = {}
            
            # Extract basic information only if explicitly mentioned
            patterns = {
                'company_name': r'(?:store|company|business)[:\s]+([^\n]+)',
                'total': r'(?:total|amount)[:\s]*\$?([\d,]+\.?\d*)',
                'date': r'(?:date)[:\s]+([^\n]+)',
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    # Only include if it's not a placeholder
                    if value and value.lower() not in ['n/a', 'unknown', 'not found', '']:
                        parsed[key] = value
            
            # Add raw response for reference
            parsed["raw_response"] = content
            
            return parsed
            
        except Exception as e:
            logger.warning(f"Failed to parse chat response: {e}")
            return {"raw_text": content}
    
    def save_results(self, results: Dict, image_path: Path) -> Path:
        """Save extraction results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract model name for filename (remove version tag if present)
        model_name = self.model.split(':')[0] if ':' in self.model else self.model
        filename = f"{image_path.stem}_{model_name}_optimized_{timestamp}.json"
        output_path = self.output_dir / filename
        
        # Add metadata
        results_with_metadata = {
            "metadata": {
                "source_file": image_path.name,
                "model_used": self.model,
                "extraction_timestamp": timestamp,
                "processing_time": results.get("processing_time", 0),
                "optimization_settings": {
                    "max_image_size": self.max_image_size,
                    "use_grayscale": self.use_grayscale,
                    "use_hybrid_ocr": self.use_hybrid_ocr,
                    "compression_quality": self.compression_quality
                }
            },
            "results": results
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def process_single_file(self, image_path: Path) -> Dict:
        """Process a single receipt file"""
        start_time = time.time()
        
        logger.info(f"Starting optimized processing of {image_path.name}")
        
        # Extract data
        results = self.extract_receipt_data(image_path)
        
        # Add processing time
        processing_time = time.time() - start_time
        results["processing_time"] = processing_time
        
        # Save results
        output_path = self.save_results(results, image_path)
        
        logger.info(f"Completed processing {image_path.name} in {processing_time:.2f}s")
        
        return {
            "input_file": image_path.name,
            "output_file": output_path.name,
            "processing_time": processing_time,
            "success": "error" not in results
        }
    
    def get_supported_files(self) -> List[Path]:
        """Get list of supported image files"""
        if not self.input_dir.exists():
            logger.error(f"Input directory {self.input_dir} does not exist")
            return []
        
        files = []
        for file_path in self.input_dir.iterdir():
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix in self.supported_formats:
                    files.append(file_path)
                elif suffix in self.pdf_formats:
                    try:
                        from pdf2image import convert_from_path
                        files.append(file_path)
                    except ImportError:
                        logger.warning(f"PDF support not available: {file_path.name}")
        
        logger.info(f"Found {len(files)} supported files in {self.input_dir}")
        return files
    
    def process_input_pattern(self, input_pattern: str) -> Dict:
        """Process files based on input pattern (single file or glob)"""
        import glob
        
        # Expand glob pattern
        files = glob.glob(input_pattern)
        
        if not files:
            logger.error(f"No files found matching pattern: {input_pattern}")
            return {"error": f"No files found matching pattern: {input_pattern}"}
        
        # Convert to Path objects and filter supported files
        file_paths = []
        for file_path in files:
            path = Path(file_path)
            if path.exists() and self._is_supported_file(path):
                file_paths.append(path)
        
        if not file_paths:
            logger.error(f"No supported files found matching pattern: {input_pattern}")
            return {"error": f"No supported files found matching pattern: {input_pattern}"}
        
        logger.info(f"Found {len(file_paths)} supported files to process")
        
        # Process each file
        results = []
        total_time = 0
        
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing {i}/{len(file_paths)}: {file_path.name}")
            
            start_time = time.time()
            result = self.process_single_file(file_path)
            processing_time = time.time() - start_time
            
            result['processing_time'] = processing_time
            results.append(result)
            total_time += processing_time
            
            # Update stats
            self.processing_stats['total_images'] += 1
            if result.get('success'):
                logger.info(f"Successfully processed {file_path.name}")
            else:
                logger.warning(f"Failed to process {file_path.name}")
        
        # Calculate averages
        if results:
            self.processing_stats['avg_time_per_image'] = total_time / len(results)
            self.processing_stats['total_time'] = total_time
        
        # Display summary
        summary = {
            'total_files': len(file_paths),
            'successful': len([r for r in results if r.get('success')]),
            'failed': len([r for r in results if not r.get('success')]),
            'total_time': total_time,
            'avg_time_per_file': total_time / len(file_paths) if file_paths else 0,
            'results': results
        }
        
        self._display_optimized_summary(summary)
        return summary

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file is supported (image or PDF)"""
        return (file_path.suffix.lower() in self.supported_formats or 
                file_path.suffix.lower() in self.pdf_formats)

    def run_bulk_scan(self) -> Dict:
        """Run optimized bulk scanning"""
        logger.info("Starting optimized bulk scan operation")
        
        files = self.get_supported_files()
        if not files:
            logger.warning("No supported files found for processing")
            return {"error": "No supported files found"}
        
        start_time = time.time()
        
        results = {
            "scan_summary": {
                "total_files": len(files),
                "processed_files": 0,
                "successful_files": 0,
                "failed_files": 0,
                "start_time": datetime.now().isoformat(),
                "model_used": self.model,
                "optimization_settings": {
                    "max_image_size": self.max_image_size,
                    "use_grayscale": self.use_grayscale,
                    "use_hybrid_ocr": self.use_hybrid_ocr,
                    "compression_quality": self.compression_quality
                }
            },
            "file_results": []
        }
        
        for i, file_path in enumerate(files, 1):
            logger.info(f"Processing file {i}/{len(files)}: {file_path.name}")
            
            try:
                file_result = self.process_single_file(file_path)
                results["file_results"].append(file_result)
                
                if file_result["success"]:
                    results["scan_summary"]["successful_files"] += 1
                else:
                    results["scan_summary"]["failed_files"] += 1
                
                results["scan_summary"]["processed_files"] += 1
                
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                results["file_results"].append({
                    "input_file": file_path.name,
                    "error": str(e),
                    "success": False
                })
                results["scan_summary"]["failed_files"] += 1
                results["scan_summary"]["processed_files"] += 1
        
        # Calculate performance statistics
        total_time = time.time() - start_time
        results["scan_summary"]["end_time"] = datetime.now().isoformat()
        results["scan_summary"]["total_processing_time"] = total_time
        results["scan_summary"]["avg_time_per_file"] = total_time / len(files) if files else 0
        
        # Add optimization statistics
        results["scan_summary"]["optimization_stats"] = {
            "traditional_ocr_success": self.processing_stats['traditional_ocr_success'],
            "vision_model_used": self.processing_stats['vision_model_used'],
            "traditional_ocr_success_rate": (self.processing_stats['traditional_ocr_success'] / len(files) * 100) if files else 0
        }
        
        # Save bulk scan summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract model name for filename (remove version tag if present)
        model_name = self.model.split(':')[0] if ':' in self.model else self.model
        summary_path = self.output_dir / f"bulk_scan_summary_{model_name}_optimized_{timestamp}.json"
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Optimized bulk scan summary saved to {summary_path}")
        except Exception as e:
            logger.error(f"Error saving bulk scan summary: {e}")
        
        # Display summary
        self._display_optimized_summary(results["scan_summary"])
        
        return results
    
    def _display_optimized_summary(self, summary: Dict):
        """Display optimized processing summary"""
        print("\n" + "="*60)
        print("OPTIMIZED BULK SCAN SUMMARY")
        print("="*60)
        
        # Handle both old and new summary formats
        if 'model_used' in summary:
            print(f"Model Used: {summary['model_used']}")
        else:
            print(f"Model Used: {self.model}")
            
        print(f"Total Files: {summary['total_files']}")
        
        if 'processed_files' in summary:
            print(f"Processed: {summary['processed_files']}")
            print(f"Successful: {summary['successful_files']}")
            print(f"Failed: {summary['failed_files']}")
            print(f"Total Time: {summary['total_processing_time']:.2f}s")
            print(f"Average Time per File: {summary['avg_time_per_file']:.2f}s")
            
            if 'optimization_stats' in summary:
                print(f"Traditional OCR Success: {summary['optimization_stats']['traditional_ocr_success']}")
                print(f"Vision Model Used: {summary['optimization_stats']['vision_model_used']}")
                print(f"Traditional OCR Success Rate: {summary['optimization_stats']['traditional_ocr_success_rate']:.1f}%")
        else:
            print(f"Successful: {summary['successful']}")
            print(f"Failed: {summary['failed']}")
            print(f"Total Time: {summary['total_time']:.2f}s")
            print(f"Average Time per File: {summary['avg_time_per_file']:.2f}s")
            
        print("="*60)

    def _process_pdf_as_image(self, pdf_path: Path) -> Dict:
        """Process PDF as image using vision model with PyMuPDF"""
        logger.info(f"Converting PDF to image for vision processing: {pdf_path.name}")
        
        try:
            import fitz  # PyMuPDF
            import cv2
            import numpy as np
            
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            
            if len(doc) == 0:
                raise ValueError("PDF has no pages")
            
            # Get first page
            page = doc[0]
            
            # Convert page to image with high resolution
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL format
            img_data = pix.tobytes("png")
            
            # Convert to OpenCV format for cropping
            nparr = np.frombuffer(img_data, np.uint8)
            opencv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Crop receipt content to remove white space
            cropped_img = self._crop_receipt_content(opencv_img)
            
            # Convert back to PIL for further processing
            cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cropped_rgb)
            
            # Apply optimizations
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # Resize if too large
            if max(pil_img.size) > self.max_image_size:
                ratio = self.max_image_size / max(pil_img.size)
                new_size = tuple(int(dim * ratio) for dim in pil_img.size)
                pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized PDF image from {pil_img.size} to {new_size}")
            
            # Convert to grayscale if enabled
            if self.use_grayscale:
                pil_img = pil_img.convert('L').convert('RGB')
            
            # Save as PNG (lossless, good for text)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG", optimize=True)
            image_data = buffered.getvalue()
            
            logger.info(f"PDF converted to PNG: {len(image_data)} bytes")
            
            # Clean up immediately
            doc.close()
            del pix
            del opencv_img
            del cropped_img
            del pil_img
            
            # Process with vision model
            self.processing_stats['vision_model_used'] += 1
            result = self._extract_with_vision_model(image_data)
            
            result["extraction_method"] = "pdf_image_conversion_pymupdf"
            result["pdf_processing"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"PDF to image processing failed: {e}")
            return {"error": f"PDF to image processing failed: {e}"}
    
    def _compare_text_vs_image_results(self, text_result: Dict, image_result: Dict) -> Dict:
        """Compare text-based vs image-based extraction results and choose the best"""
        logger.info("Comparing text-based vs image-based extraction results")
        
        # Define comparison criteria
        text_score = 0
        image_score = 0
        
        # Compare data richness (more fields = better)
        text_fields = len([k for k, v in text_result.get("parsed_data", {}).items() 
                          if v and str(v).strip()])
        image_fields = len([k for k, v in image_result.get("parsed_data", {}).items() 
                           if v and str(v).strip()])
        
        if text_fields > image_fields:
            text_score += 2
        elif image_fields > text_fields:
            image_score += 2
        
        # Compare processing time (faster = better)
        text_time = text_result.get("processing_time", 0)
        image_time = image_result.get("processing_time", 0)
        
        if text_time < image_time:
            text_score += 1
        elif image_time < text_time:
            image_score += 1
        
        # Compare confidence levels
        text_confidence = text_result.get("confidence", "low")
        image_confidence = image_result.get("confidence", "low")
        
        confidence_scores = {"high": 3, "medium": 2, "low": 1}
        text_score += confidence_scores.get(text_confidence, 1)
        image_score += confidence_scores.get(image_confidence, 1)
        
        # Determine winner
        if text_score > image_score:
            winner = "text"
            logger.info(f"Text-based extraction chosen (score: {text_score} vs {image_score})")
        elif image_score > text_score:
            winner = "image"
            logger.info(f"Image-based extraction chosen (score: {image_score} vs {text_score})")
        else:
            winner = "text"  # Prefer text for tie (usually faster)
            logger.info(f"Tie - choosing text-based extraction (score: {text_score})")
        
        # Return comparison result
        return {
            "winner": winner,
            "text_result": text_result,
            "image_result": image_result,
            "comparison_scores": {
                "text_score": text_score,
                "image_score": image_score,
                "text_fields": text_fields,
                "image_fields": image_fields,
                "text_time": text_time,
                "image_time": image_time,
                "text_confidence": text_confidence,
                "image_confidence": image_confidence
            },
            "extraction_method": f"pdf_{winner}_based",
            "pdf_processing": True
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Optimized Receipt OCR Scanner for Resource-Constrained Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python receipt_ocr_scanner.py --input Receipts/french.pdf
  python receipt_ocr_scanner.py --input "Receipts/*" --model llava:latest
  python receipt_ocr_scanner.py --input "Receipts/*.pdf" --max-image-size 600
  python receipt_ocr_scanner.py --input "Receipts/*" --no-hybrid-ocr
        """
    )
    
    parser.add_argument(
        '--model',
        default='llava:7b',
        help='Vision model to use (default: llava:7b - lighter and faster)'
    )
    
    parser.add_argument(
        '--input',
        default='./Receipts/*',
        help='Input file or glob pattern (default: ./Receipts/* for all files)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./Reports',
        help='Directory to save results (default: ./Reports)'
    )
    
    parser.add_argument(
        '--max-image-size',
        type=int,
        default=800,
        help='Maximum image dimension for resizing (default: 800)'
    )
    
    parser.add_argument(
        '--use-grayscale',
        action='store_true',
        default=True,
        help='Convert images to grayscale for faster processing (default: True)'
    )
    
    parser.add_argument(
        '--no-grayscale',
        dest='use_grayscale',
        action='store_false',
        help='Disable grayscale conversion'
    )
    
    parser.add_argument(
        '--hybrid-ocr',
        action='store_true',
        default=True,
        help='Use traditional OCR first, then vision model if needed (default: True)'
    )
    
    parser.add_argument(
        '--no-hybrid-ocr',
        dest='use_hybrid_ocr',
        action='store_false',
        help='Disable hybrid OCR approach'
    )
    
    parser.add_argument(
        '--compression-quality',
        type=int,
        default=85,
        help='JPEG compression quality (1-100, default: 85)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize optimized scanner
    scanner = OptimizedReceiptOCRScanner(
        input_dir=args.output_dir,  # Use output_dir as input_dir for compatibility
        output_dir=args.output_dir,
        model=args.model,
        max_image_size=args.max_image_size,
        use_grayscale=args.use_grayscale,
        use_hybrid_ocr=args.use_hybrid_ocr,
        compression_quality=args.compression_quality
    )
    
    try:
        # Process input pattern
        scanner.process_input_pattern(args.input)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
