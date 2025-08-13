#!/usr/bin/env python3
"""
Optimized Receipt OCR Scanner - Byte-buffer (in-memory) version
All image/PDF processing uses bytes instead of file paths so you can
wire it directly to an API upload or array buffer.
"""

import sys
import argparse
import json
import time
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
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

# Optional: OpenCV
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: opencv-python not available. Install with: pip install opencv-python")

# Optional: PyMuPDF & PyPDF2 for PDFs
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("receipt_ocr_optimized.log")]
)
logger = logging.getLogger(__name__)


def _looks_like_pdf(data: bytes) -> bool:
    # PDFs start with %PDF
    return len(data) >= 4 and data[:4] == b"%PDF"


class OptimizedReceiptOCRScanner:
    """Optimized Receipt OCR Scanner (buffer-based)"""

    def __init__(
        self,
        output_dir: str = "Reports",
        model: str = "llava:7b",          # Use lighter model by default
        max_image_size: int = 800,        # Max dimension for resizing
        use_grayscale: bool = True,       # Convert to grayscale for text
        use_hybrid_ocr: bool = True,      # Use traditional OCR first
        compression_quality: int = 85     # JPEG compression quality
    ):
        self.output_dir = Path(output_dir)
        self.model = model
        self.max_image_size = max_image_size
        self.use_grayscale = use_grayscale
        self.use_hybrid_ocr = use_hybrid_ocr and TRADITIONAL_OCR_AVAILABLE
        self.compression_quality = compression_quality

        self.output_dir.mkdir(exist_ok=True)

        # Load template for structured output
        self.template = self._load_template()

        logger.info(f"Checking compatibility for model: {model}")
        if not self._is_vision_model():
            logger.warning(f"Model {model} may not be a vision model")
            if not self._test_model_compatibility():
                logger.warning(f"Model {model} failed vision compatibility test")
                fallback = self._get_fallback_model()
                if fallback != model:
                    logger.info(f"Suggesting fallback model: {fallback}")

        # Performance tracking
        self.processing_stats = {
            "traditional_ocr_success": 0,
            "vision_model_used": 0,
        }

        logger.info(f"Initialized OptimizedReceiptOCRScanner with model: {model}")
        logger.info(f"Max image size: {max_image_size}px, Grayscale: {use_grayscale}")
        logger.info(f"Hybrid OCR: {self.use_hybrid_ocr}")

    # -------------------- Templates / prompts --------------------

    def _load_template(self) -> Dict:
        """Load the receipt template for structured output (optional)"""
        template_path = Path("Template/llama3.2-vision_template.json")
        if template_path.exists():
            try:
                with open(template_path, "r", encoding="utf-8") as f:
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

    def _get_optimized_prompt(self) -> str:
        return (
            "Analyze this receipt image and extract the key information.\n\n"
            "Please provide the following information if visible:\n"
            "- Store/Company name\n- Date and time\n- Items with prices (if listed)\n"
            "- Subtotal\n- Tax amount\n- Total amount\n- Payment method\n\n"
            "If any information is not visible or unclear, please indicate that.\n\n"
            "Format your response clearly and be specific about what you can see in the image."
        )

    # -------------------- Buffer-based image/PDF utilities --------------------

    def _optimize_image_bytes(self, image_bytes: bytes) -> bytes:
        """Optimize image bytes for faster processing (resize/grayscale/compress)."""
        try:
            if not TRADITIONAL_OCR_AVAILABLE:
                # No PIL; just return original bytes
                return image_bytes

            with Image.open(io.BytesIO(image_bytes)) as img:
                # Normalize to RGB
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize if too large
                if max(img.size) > self.max_image_size:
                    ratio = self.max_image_size / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"Resized image to {new_size}")

                # Grayscale if configured (helps OCR/LLM on text receipts)
                if self.use_grayscale:
                    img = img.convert("L").convert("RGB")

                # Save with compression
                buffered = io.BytesIO()
                img.save(
                    buffered,
                    format="JPEG",
                    quality=self.compression_quality,
                    optimize=True
                )
                return buffered.getvalue()

        except Exception as e:
            logger.error(f"Error optimizing image bytes: {e}")
            # Fall back to original bytes
            return image_bytes

    def _crop_receipt_content(self, image_bgr: "np.ndarray") -> "np.ndarray":
        """Crop receipt content using OpenCV - Conservative approach."""
        if not OPENCV_AVAILABLE:
            return image_bgr

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image_bgr

        significant = [c for c in contours if cv2.contourArea(c) > 1000]
        if not significant:
            return image_bgr

        x, y, w, h = cv2.boundingRect(max(significant, key=cv2.contourArea))
        padding = 50
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image_bgr.shape[1] - x, w + 2 * padding)
        h = min(image_bgr.shape[0] - y, h + 2 * padding)
        cropped = image_bgr[y:y + h, x:x + w]
        logger.info(f"Cropped to {cropped.shape[1]}x{cropped.shape[0]}")
        return cropped

    # -------------------- PDF handling (bytes) --------------------

    def _analyze_pdf_content_bytes(self, pdf_bytes: bytes) -> Dict:
        """Analyze PDF content (bytes) to see if it contains text or is image-based."""
        if not PYPDF2_AVAILABLE:
            return {"type": "image", "pages": 1, "note": "PyPDF2 not available"}

        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            if len(reader.pages) == 0:
                return {"type": "empty", "pages": 0}
            first = reader.pages[0]
            text = first.extract_text() or ""
            if text.strip() and len(text) > 50:
                return {
                    "type": "text",
                    "pages": len(reader.pages),
                    "text_length": len(text),
                    "sample_text": (text[:200] + "...") if len(text) > 200 else text
                }
            return {"type": "image", "pages": len(reader.pages), "text_length": len(text)}
        except Exception as e:
            logger.warning(f"Could not analyze PDF bytes: {e}")
            return {"type": "image", "pages": 1, "error": str(e)}

    def _extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> Dict:
        """Extract text from a text-based PDF (bytes)."""
        if not PYPDF2_AVAILABLE:
            return {"error": "PDF text extraction unavailable (PyPDF2 not installed)"}
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text_content = ""
            for i, page in enumerate(reader.pages, 1):
                page_text = page.extract_text() or ""
                if page_text:
                    text_content += f"\n--- Page {i} ---\n{page_text}"
            if not text_content.strip():
                return {"error": "No text content found in PDF"}
            parsed = self._parse_ocr_text(text_content)
            return {
                "raw_text": text_content,
                "parsed_data": parsed,
                "extraction_method": "pdf_text_extraction",
                "confidence": "high" if len(text_content.strip()) > 100 else "low",
            }
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return {"error": f"PDF text extraction failed: {e}"}

    def _pdf_to_image_bytes(self, pdf_bytes: bytes) -> Optional[bytes]:
        """Render first page of PDF (bytes) to a cropped PNG (bytes)."""
        if not (PYMUPDF_AVAILABLE and OPENCV_AVAILABLE and TRADITIONAL_OCR_AVAILABLE):
            # Minimal fallback: return None to skip image conversion
            return None
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if len(doc) == 0:
                return None
            page = doc[0]
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_png = pix.tobytes("png")
            nparr = np.frombuffer(img_png, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cropped = self._crop_receipt_content(img_bgr)

            # Convert back to PIL and optimize similarly
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            if max(pil_img.size) > self.max_image_size:
                ratio = self.max_image_size / max(pil_img.size)
                new_size = (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio))
                pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            if self.use_grayscale:
                pil_img = pil_img.convert("L").convert("RGB")

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
        except Exception as e:
            logger.error(f"PDF to image conversion failed: {e}")
            return None

    # -------------------- OCR & LLM extraction (bytes) --------------------

    def _extract_with_traditional_ocr_bytes(self, image_bytes: bytes) -> Dict:
        """Extract text using pytesseract from image bytes."""
        if not TRADITIONAL_OCR_AVAILABLE:
            return {"error": "Traditional OCR not available"}

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if max(img.size) > 1200:
                    ratio = 1200 / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                text = pytesseract.image_to_string(img)

            if not text.strip():
                return {"error": "No text extracted", "method": "traditional_ocr"}

            parsed = self._parse_ocr_text(text)
            return {
                "raw_text": text,
                "parsed_data": parsed,
                "extraction_method": "traditional_ocr",
                "confidence": "high" if len(text.strip()) > 50 else "low",
            }
        except Exception as e:
            logger.error(f"Traditional OCR failed: {e}")
            return {"error": f"Traditional OCR failed: {e}", "method": "traditional_ocr"}

    def _extract_with_vision_model(self, image_data: bytes) -> Dict:
        """Extract data using a vision model (Ollama) from image bytes."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": self._get_optimized_prompt(),
                    "images": [image_data],
                }]
            )
            content = response.get("message", {}).get("content", "")
            if not content:
                return {"error": "No response content from vision model"}
            parsed = self._parse_chat_response(content)
            return {
                "raw_response": content,
                "parsed_data": parsed,
                "extraction_method": "vision_model",
                "model_used": self.model,
            }
        except Exception as e:
            logger.error(f"Vision model extraction failed for {self.model}: {e}")
            msg = str(e)
            if "tool" in msg.lower():
                return {"error": f"Tool support issue: {msg}"}
            if "image" in msg.lower():
                return {"error": f"Image support issue: {msg}"}
            return {"error": f"Vision model failed: {msg}"}

    # -------------------- Parsing helpers --------------------

    def _parse_ocr_text(self, text: str) -> dict:
        """
        Parse OCR text for receipts with a 'Description / Qty / Unit Price / Total' table
        followed by a summary block with Subtotal / Tax / Total.

        - Extracts Company Name (first plausible header line)
        - Parses items as 4-line blocks (desc, qty, unit, total)
        - Extracts Subtotal, Sales Tax (or VAT), and Total separately
        - If Total is missing or equals Subtotal while Tax exists, computes Total = Subtotal + Tax
        """
        import re

        # ---------- helpers ----------
        MONEY = r"\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})"

        def is_money(s: str) -> bool:
            return bool(re.fullmatch(MONEY, s.strip()))

        def norm_money(s: str) -> str:
            s = s.strip()
            return s if s.startswith("$") else f"${s}" if is_money(s) else s

        def to_float(m: str) -> float:
            return float(m.replace("$", "").replace(",", ""))

        def is_qty(s: str) -> bool:
            return bool(re.fullmatch(r"\d+", s.strip()))

        def is_col_header(s: str) -> bool:
            s = s.strip().lower()
            return s in {"description", "qty", "quantity", "unit price", "price", "total"}

        def next_nonempty_amount(idx: int) -> str | None:
            j = idx + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and is_money(lines[j]):
                return norm_money(lines[j])
            return None

        # ---------- prep ----------
        lines = [ln.strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]  # drop blanks
        low_lines = [ln.lower() for ln in lines]

        # ---------- Company Name ----------
        company = ""
        for ln in lines[:8]:
            if ln.startswith("---"):            # page marker
                continue
            if ":" in ln:                       # meta fields like "Phone: ..."
                continue
            if is_col_header(ln):               # table headers
                continue
            if re.search(r"receipt|invoice|statement", ln, re.IGNORECASE):
                continue
            if is_money(ln) or is_qty(ln):      # not a numeric/price-only line
                continue
            company = ln
            break

        # ---------- Date ----------
        date_val = None
        # Prefer explicit meta like "Check-in Date: 2025-08-01"
        for ln in lines:
            m = re.search(r"(check-?in|check in)\s*date[:\s]+(.+)$", ln, re.IGNORECASE)
            if m:
                m2 = re.search(r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})|(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", m.group(2))
                if m2:
                    date_val = m2.group()
                    break
        # Else first date-like token anywhere
        if not date_val:
            for ln in lines:
                m = re.search(r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})|(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", ln)
                if m:
                    date_val = m.group()
                    break

        # ---------- Locate table header (Description / ... / Total) ----------
        header_idx = None
        for i in range(len(lines)):
            win = {lines[i].lower()}
            if i + 1 < len(lines): win.add(lines[i+1].lower())
            if i + 2 < len(lines): win.add(lines[i+2].lower())
            if i + 3 < len(lines): win.add(lines[i+3].lower())
            if {"description", "total"}.issubset({w.strip().lower() for w in win}):
                header_idx = i
                break

        # ---------- Items (parse as 4-line blocks) ----------
        items = []
        items_end_idx = None
        if header_idx is not None:
            j = header_idx
            while j < len(lines) and is_col_header(lines[j]):
                j += 1

            while j + 3 < len(lines):
                # Stop if entering a summary label
                if re.search(r"^\s*(sub\s*total|subtotal|tax|vat|total|grand\s+total)\b", lines[j], re.IGNORECASE):
                    items_end_idx = j
                    break

                desc, qty, unit, tot = lines[j], lines[j+1], lines[j+2], lines[j+3]
                desc_ok = (not is_col_header(desc)) and (":" not in desc) and (not is_money(desc)) and (not is_qty(desc))
                if desc_ok and is_qty(qty) and is_money(unit) and is_money(tot):
                    items.append({
                        "description": desc,
                        "quantity": qty,
                        "unit_price": norm_money(unit),
                        "total": norm_money(tot),
                        "category": ""
                    })
                    j += 4
                else:
                    j += 1  # resync
            if items_end_idx is None:
                items_end_idx = j  # if we ran out

        # ---------- Summary block (Subtotal / Tax / Total) ----------
        # Only search **after** items_end_idx to avoid the "Total" column header.
        start_summary = items_end_idx if items_end_idx is not None else 0

        subtotal_val = None
        tax_val = None
        total_val = None

        for i in range(start_summary, len(lines)):
            ln = lines[i]
            low = low_lines[i]

            # Subtotal
            if subtotal_val is None and re.search(r"\bsub\s*total\b|^subtotal\b", low):
                # amount on same line or next non-empty line
                m = re.search(MONEY + r"\s*$", ln)
                if m:
                    subtotal_val = norm_money(m.group())
                else:
                    maybe = next_nonempty_amount(i)
                    if maybe: subtotal_val = maybe
                continue

            # Tax / VAT (capture first; you can extend to collect multiple)
            if tax_val is None and re.search(r"\b(tax|vat)\b", low) and not re.search(r"\bno\s*tax\b", low):
                m = re.search(MONEY + r"\s*$", ln)
                if m:
                    tax_val = norm_money(m.group())
                else:
                    maybe = next_nonempty_amount(i)
                    if maybe: tax_val = maybe
                continue

            # Total / Grand Total (explicit summary, not the table header)
            if re.fullmatch(r"(?i)(?:grand\s+)?total[:\s]*.*", ln):
                m = re.search(MONEY + r"\s*$", ln)
                if m:
                    total_val = norm_money(m.group())
                else:
                    maybe = next_nonempty_amount(i)
                    if maybe: total_val = maybe
                continue

        # Fallbacks / corrections
        if total_val is None and subtotal_val and tax_val:
            # If total not present, compute it
            total_val = f"${to_float(subtotal_val) + to_float(tax_val):.2f}"

        # Guard against "Total == Subtotal" when Tax exists (table-header confusion)
        if total_val and subtotal_val and tax_val:
            if abs(to_float(total_val) - to_float(subtotal_val)) < 0.005:
                total_val = f"${to_float(subtotal_val) + to_float(tax_val):.2f}"

        # As a last resort (no explicit summary), pick the maximum money AFTER the items section
        if total_val is None:
            monies_after = re.findall(MONEY, "\n".join(lines[start_summary:]))
            if monies_after:
                total_val = norm_money(max(monies_after, key=lambda s: to_float(s)))

        # ---------- Build result ----------
        parsed: dict = {}
        if company: parsed["Company Name"] = company
        if date_val: parsed["Date"] = date_val
        if items: parsed["Items"] = items
        if subtotal_val: parsed["Subtotal"] = subtotal_val
        if tax_val: parsed["Sales Tax"] = tax_val   # matches your template key
        if total_val: parsed["Total"] = total_val

        return parsed


    def _parse_chat_response(self, content: str) -> Dict:
        """Try to pull structured fields from the model's text response."""
        try:
            import re
            # Try JSON
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                try:
                    j = json.loads(m.group())
                    cleaned = {k: v for k, v in j.items()
                               if str(v).strip() and str(v).lower() not in ["n/a", "unknown", "not found"]}
                    if cleaned:
                        return cleaned
                except json.JSONDecodeError:
                    pass

            parsed = {}
            patterns = {
                "company_name": r"(?:store|company|business)[:\s]+([^\n]+)",
                "total": r"(?:total|amount)[:\s]*\$?([\d,]+\.?\d*)",
                "date": r"(?:date)[:\s]+([^\n]+)",
            }
            for key, pat in patterns.items():
                mm = re.search(pat, content, re.IGNORECASE)
                if mm:
                    val = mm.group(1).strip()
                    if val and val.lower() not in ["n/a", "unknown", "not found"]:
                        parsed[key] = val
            parsed["raw_response"] = content
            return parsed
        except Exception as e:
            logger.warning(f"Failed to parse chat response: {e}")
            return {"raw_text": content}

    # -------------------- Public API (bytes) --------------------

    def extract_receipt_data_from_bytes(self, data: bytes, source_name: str = "upload") -> Dict:
        """
        Main entrypoint: pass raw bytes of an image or a PDF.
        Returns structured extraction results.
        """
        start_time = time.time()
        try:
            if _looks_like_pdf(data):
                logger.info(f"Processing PDF bytes for {source_name}")
                pdf_analysis = self._analyze_pdf_content_bytes(data)
                logger.info(f"PDF analysis: {pdf_analysis}")

                if pdf_analysis.get("type") == "text":
                    # Prefer direct text extraction but also compare against image render
                    text_result = self._extract_text_from_pdf_bytes(data)
                    if "error" not in text_result:
                        image_bytes = self._pdf_to_image_bytes(data)
                        if image_bytes:
                            image_result = self._process_image_bytes(image_bytes)
                            comparison = self._compare_text_vs_image_results(text_result, image_result)
                            comparison["processing_time"] = time.time() - start_time
                            comparison["pdf_processing"] = True
                            comparison["pdf_analysis"] = pdf_analysis
                            return comparison
                        # Only text path succeeded
                        text_result["processing_time"] = time.time() - start_time
                        return text_result
                    else:
                        # Text failed; try image path
                        image_bytes = self._pdf_to_image_bytes(data)
                        if not image_bytes:
                            return {"error": "Failed to process PDF as image and text extraction failed"}
                        return self._process_image_bytes(image_bytes)
                else:
                    # Image-based PDF
                    image_bytes = self._pdf_to_image_bytes(data)
                    if not image_bytes:
                        return {"error": "PDF seems image-based but conversion failed"}
                    result = self._process_image_bytes(image_bytes)
                    result["pdf_processing"] = True
                    return result
            else:
                # Regular image
                return self._process_image_bytes(data)
        except Exception as e:
            logger.error(f"Error processing buffer for {source_name}: {e}")
            return {
                "error": str(e),
                "source": source_name,
                "timestamp": datetime.now().isoformat(),
                "processing_time": time.time() - start_time
            }

    def _process_image_bytes(self, image_bytes: bytes) -> Dict:
        """Run hybrid OCR then LLM vision on raw image bytes."""
        start_time = time.time()

        # Try traditional OCR (fast path)
        if self.use_hybrid_ocr:
            logger.info("Attempting traditional OCR first...")
            ocr_result = self._extract_with_traditional_ocr_bytes(image_bytes)
            if "error" not in ocr_result and ocr_result.get("confidence") == "high":
                self.processing_stats["traditional_ocr_success"] += 1
                ocr_result["processing_time"] = time.time() - start_time
                logger.info(f"Traditional OCR successful in {ocr_result['processing_time']:.2f}s")
                return ocr_result

        # Vision model (optimize image first)
        logger.info("Using vision model for extraction...")
        self.processing_stats["vision_model_used"] += 1

        if not self._test_model_compatibility():
            logger.warning(f"Model {self.model} may not support vision input")
            fallback = self._get_fallback_model()
            if fallback != self.model:
                logger.info(f"Switching to fallback model: {fallback}")
                original = self.model
                self.model = fallback
                if not self._test_model_compatibility():
                    logger.error("Fallback model also failed vision test")
                    self.model = original
                    return {"error": "No compatible vision model available"}

        optimized = self._optimize_image_bytes(image_bytes)
        result = self._extract_with_vision_model(optimized)
        result["processing_time"] = time.time() - start_time
        logger.info(f"Vision model processing completed in {result['processing_time']:.2f}s")
        return result

    # -------------------- Result persistence --------------------

    def save_results(self, results: Dict, source_name: str = "upload") -> Path:
        """Save extraction results to file (source_name is used for metadata & filename)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model.split(":")[0] if ":" in self.model else self.model
        safe_name = (
            source_name.replace("/", "_").replace("\\", "_").replace(" ", "_")[:40]
            or "upload"
        )
        filename = f"{safe_name}_{model_name}_optimized_{timestamp}.json"
        output_path = self.output_dir / filename

        results_with_metadata = {
            "metadata": {
                "source_name": source_name,
                "model_used": self.model,
                "extraction_timestamp": timestamp,
                "processing_time": results.get("processing_time", 0),
                "optimization_settings": {
                    "max_image_size": self.max_image_size,
                    "use_grayscale": self.use_grayscale,
                    "use_hybrid_ocr": self.use_hybrid_ocr,
                    "compression_quality": self.compression_quality,
                },
            },
            "results": results,
        }

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    # -------------------- Comparison (text vs image) --------------------

    def _compare_text_vs_image_results(self, text_result: Dict, image_result: Dict) -> Dict:
        logger.info("Comparing text-based vs image-based extraction results")
        text_score = 0
        image_score = 0

        def score_fields(d: Dict) -> int:
            return len([k for k, v in d.get("parsed_data", {}).items() if str(v).strip()])

        tf = score_fields(text_result)
        imf = score_fields(image_result)
        if tf > imf:
            text_score += 2
        elif imf > tf:
            image_score += 2

        tt = text_result.get("processing_time", 0)
        it = image_result.get("processing_time", 0)
        if tt < it:
            text_score += 1
        elif it < tt:
            image_score += 1

        conf_map = {"high": 3, "medium": 2, "low": 1}
        text_score += conf_map.get(text_result.get("confidence", "low"), 1)
        image_score += conf_map.get(image_result.get("confidence", "low"), 1)

        if text_score > image_score:
            winner = "text"
        elif image_score > text_score:
            winner = "image"
        else:
            winner = "text"  # prefer faster path on tie

        return {
            "winner": winner,
            "text_result": text_result,
            "image_result": image_result,
            "comparison_scores": {
                "text_score": text_score,
                "image_score": image_score,
                "text_fields": tf,
                "image_fields": imf,
                "text_time": tt,
                "image_time": it,
                "text_confidence": text_result.get("confidence", "low"),
                "image_confidence": image_result.get("confidence", "low"),
            },
            "extraction_method": f"pdf_{winner}_based",
            "pdf_processing": True,
        }

    # -------------------- Model capability helpers --------------------

    def _get_model_info(self) -> Dict:
        try:
            info = ollama.show(self.model)
            return {
                "name": info.get("name", self.model),
                "family": info.get("family", "unknown"),
                "parameter_size": info.get("parameter_size", "unknown"),
                "modelfile": info.get("modelfile", ""),
            }
        except Exception as e:
            logger.warning(f"Could not get model info for {self.model}: {e}")
            return {"name": self.model, "family": "unknown", "parameter_size": "unknown", "modelfile": ""}

    def _is_vision_model(self) -> bool:
        mf = self._get_model_info().get("modelfile", "").lower()
        return any(k in mf for k in ["vision", "llava", "clip", "image", "multimodal"])

    def _test_model_compatibility(self) -> bool:
        try:
            test_image = (
                b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
                b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
                b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f"
                b"\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
                b"\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01"
                b"\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00"
                b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01"
                b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                b"\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xaa\xff\xd9"
            )
            ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": "Test message", "images": [test_image]}]
            )
            return True
        except Exception as e:
            logger.warning(f"Model {self.model} may not support vision: {e}")
            return False

    def _get_fallback_model(self) -> str:
        try:
            available = [m["name"] for m in ollama.list()["models"]]
        except Exception as e:
            logger.warning(f"Could not get available models: {e}")
            available = []
        preferred = ["llava:latest", "minicpm-v:latest", "granite3.2-vision:latest"]
        for m in preferred:
            if m in available and m != self.model:
                return m
        for m in available:
            if m != self.model:
                return m
        return self.model

# -------------------- CLI (still in-memory) --------------------

def main():
    """
    Example CLI for local testing.
    In production/API: call `extract_receipt_data_from_bytes(upload_bytes, source_name)`
    and `save_results(...)` if you want to persist the output.
    """
    parser = argparse.ArgumentParser(
        description="Optimized Receipt OCR Scanner (buffer-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python receipt_ocr_scanner.py --input ./Receipts/sample.jpg
  python receipt_ocr_scanner.py --input ./Receipts/sample.pdf --model llava:latest
        """
    )

    parser.add_argument("--model", default="llava:7b", help="Vision model to use")
    parser.add_argument("--input", required=True, help="Path to a single file (image or PDF)")
    parser.add_argument("--output-dir", default="./Reports", help="Where to save JSON results")
    parser.add_argument("--max-image-size", type=int, default=800, help="Max dimension for resizing")
    parser.add_argument("--use-grayscale", action="store_true", default=True, help="Enable grayscale")
    parser.add_argument("--no-grayscale", dest="use_grayscale", action="store_false", help="Disable grayscale")
    parser.add_argument("--hybrid-ocr", action="store_true", default=True, help="Use pytesseract first")
    parser.add_argument("--no-hybrid-ocr", dest="use_hybrid_ocr", action="store_false", help="Disable pytesseract")
    parser.add_argument("--compression-quality", type=int, default=85, help="JPEG quality 1-100")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    scanner = OptimizedReceiptOCRScanner(
        output_dir=args.output_dir,
        model=args.model,
        max_image_size=args.max_image_size,
        use_grayscale=args.use_grayscale,
        use_hybrid_ocr=args.use_hybrid_ocr,
        compression_quality=args.compression_quality
    )

    # Read file into memory once; run the same byte-oriented path used by the API
    path = Path(args.input)
    if not path.exists() or not path.is_file():
        logger.error(f"Input file not found: {path}")
        sys.exit(1)
    data = path.read_bytes()
    results = scanner.extract_receipt_data_from_bytes(data, source_name=path.name)
    out = scanner.save_results(results, source_name=path.name)
    print(f"Saved results: {out}")

if __name__ == "__main__":
    main()
