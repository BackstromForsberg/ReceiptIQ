#!/usr/bin/env python3
"""
Receipt OCR Scanner (buffer-based; robust text-PDF vs image handling)
- Accepts raw bytes, base64 strings, or data URLs
- Detects PDF vs image by content, not headers
- Text-based PDFs -> PyPDF2 text extraction (no vision)
- Image-based PDFs/images -> Vision model (Ollama) with optional hybrid pytesseract first
"""

import os, sys, io, re, json, time, base64, socket, logging, imghdr
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple

# ---------- Optional deps ----------
try:
    import pytesseract
    from PIL import Image
    TRAD_OCR = True
except Exception:
    TRAD_OCR = False
    print("Warning: pytesseract not available. `pip install pytesseract pillow` and install `tesseract-ocr` binary.")

try:
    import cv2
    import numpy as np
    OPENCV = True
except Exception:
    OPENCV = False
    print("Warning: opencv-python not available. `pip install opencv-python` to enable cropping.")

try:
    import fitz  # PyMuPDF
    PYMUPDF = True
except Exception:
    PYMUPDF = False
    print("Warning: PyMuPDF not available. `pip install PyMuPDF` for PDF rendering.")

try:
    import PyPDF2
    PYPDF2 = True
except Exception:
    PYPDF2 = False
    print("Warning: PyPDF2 not available. `pip install PyPDF2` for text-PDF extraction.")

# ---------- Ollama + HTTP ----------
try:
    import requests
except Exception:
    print("Error: requests not found. `pip install requests`")
    sys.exit(1)

try:
    import ollama
except Exception:
    print("Error: ollama not found. `pip install ollama`")
    sys.exit(1)

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("receipt_ocr_optimized.log")]
)
log = logging.getLogger(__name__)

# ---------- Helpers ----------
DATA_URL_RE = re.compile(r'^data:([-\w]+/[-\w+.]+);base64,')

def _env_ollama_host() -> str:
    url = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
    if url.startswith("//"): url = "http:" + url
    if not url.startswith(("http://", "https://")): url = "http://" + url
    return url.rstrip("/")

def _split_host_port(url: str) -> Tuple[str, int]:
    h = url.split("://", 1)[1]
    h = h.split("/", 1)[0] if "/" in h else h
    host, _, port = h.partition(":")
    return host, int(port or "11434")

def _wait_dns(host: str, timeout_s: float = 30):
    end = time.time() + timeout_s
    while True:
        try:
            socket.getaddrinfo(host, None)
            return
        except socket.gaierror:
            if time.time() >= end:
                raise RuntimeError(f"DNS resolve failed for '{host}' after {timeout_s}s")
            time.sleep(0.5)

def _wait_http(url: str, timeout_s: float = 120):
    end = time.time() + timeout_s
    tags = f"{url}/api/tags"
    while True:
        try:
            r = requests.get(tags, timeout=2)
            if r.ok:
                return
        except Exception:
            pass
        if time.time() >= end:
            raise RuntimeError(f"Ollama not responding at {tags} after {timeout_s}s")
        time.sleep(0.5)

def _ensure_model(url: str, model: str, timeout_s: float = 900):
    """Pull model if missing; stream status, fail on error/timeout."""
    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        if resp.ok and model in [m.get("name") for m in resp.json().get("models", [])]:
            return
    except Exception:
        pass
    log.info(f"Pulling model '{model}' ...")
    s = requests.Session()
    r = s.post(f"{url}/api/pull", json={"name": model}, stream=True, timeout=15)
    r.raise_for_status()
    start = time.time()
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        if "error" in line.lower():
            raise RuntimeError(f"ollama pull error: {line}")
        if '"done":true' in line or '"status":"success"' in line or '"status":"downloaded"' in line:
            break
        if time.time() - start > timeout_s:
            raise RuntimeError(f"Timed out pulling '{model}' after {timeout_s}s")
    log.info(f"Model '{model}' ready")

_OLLAMA_CLIENT = None
def _ollama_client(ensure: Optional[str] = None):
    """Lazy client; waits for DNS/HTTP; pulls model if missing."""
    global _OLLAMA_CLIENT
    if _OLLAMA_CLIENT is not None:
        return _OLLAMA_CLIENT
    url = _env_ollama_host()
    host, _ = _split_host_port(url)
    log.info(f"Ollama target: {url}")
    _wait_dns(host, 30)
    _wait_http(url, 120)
    if ensure:
        _ensure_model(url, ensure, 900)
    _OLLAMA_CLIENT = ollama.Client(host=url)
    return _OLLAMA_CLIENT

def _normalize_to_bytes(x) -> bytes:
    """
    Accept:
      - bytes/bytearray
      - base64 string
      - data URL string: data:<mime>;base64,<data>
    Return bytes; raise on unsupported.
    """
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, str):
        s = x.strip()
        m = DATA_URL_RE.match(s[:100])
        if m:
            try:
                return base64.b64decode(s[m.end():], validate=True)
            except Exception as e:
                raise TypeError(f"Invalid data URL base64: {e}") from e
        # try bare base64
        try:
            return base64.b64decode(s, validate=True)
        except Exception:
            raise TypeError("Expected raw bytes, base64 string, or data URL; got plain str.")
    raise TypeError(f"Unsupported type for payload: {type(x)}")

def _detect_payload_type(b: bytes) -> str:
    """Return 'pdf' if starts with %PDF, 'image' for known images (or PIL-loadable), else 'unknown'."""
    if len(b) >= 4 and b[:4] == b"%PDF":
        return "pdf"
    kind = imghdr.what(None, b)
    if kind in {"png", "jpeg", "jpg", "bmp", "gif", "tiff"}:
        return "image"
    if TRAD_OCR:
        try:
            Image.open(io.BytesIO(b))
            return "image"
        except Exception:
            pass
    return "unknown"

# ---------- Scanner ----------
class OptimizedReceiptOCRScanner:
    def __init__(
        self,
        output_dir: str = "Reports",
        model: str = "llava:7b",
        max_image_size: int = 800,
        use_grayscale: bool = True,
        use_hybrid_ocr: bool = True,
        compression_quality: int = 85,
    ):
        self.output_dir = Path(output_dir); self.output_dir.mkdir(exist_ok=True)
        self.model = model
        self.max_image_size = max_image_size
        self.use_grayscale = use_grayscale
        self.use_hybrid_ocr = use_hybrid_ocr and TRAD_OCR
        self.compression_quality = compression_quality
        self.template = self._load_template()
        self.processing_stats = {"traditional_ocr_success": 0, "vision_model_used": 0}
        log.info(f"Initialized OptimizedReceiptOCRScanner model={model} max={max_image_size} gray={use_grayscale} hybrid={self.use_hybrid_ocr}")

    # ----- Template / Prompt -----
    def _load_template(self) -> Dict:
        p = Path("Template/llama3.2-vision_template.json")
        if p.exists():
            try:
                return json.load(open(p, "r", encoding="utf-8"))
            except Exception as e:
                log.warning(f"Could not load template: {e}")
        return {
            "Company Name": "", "Receipt Number": "", "Date": "", "Time": "",
            "Cashier": "", "Store Location": "", "Items": [],
            "Subtotal": "", "Sales Tax": "", "Total": "", "Payment Method": "", "Change": "",
            "Processing Notes": {"Image Quality": "", "Text Clarity": "", "Layout Complexity": "", "Extraction Confidence": ""}
        }

    def _prompt(self) -> str:
        return (
            "Analyze this receipt image and extract:\n"
            "- Store/Company name\n- Date and time\n- Item lines (description, qty, unit price, line total)\n"
            "- Subtotal\n- Tax amount\n- Total amount\n- Payment method\n"
            "If something is missing/unclear, say so. Be specific."
        )

    # ----- Imaging -----
    def _optimize_image_bytes(self, b: bytes) -> bytes:
        if not TRAD_OCR:
            return b
        try:
            with Image.open(io.BytesIO(b)) as img:
                if img.mode != "RGB": img = img.convert("RGB")
                if max(img.size) > self.max_image_size:
                    r = self.max_image_size / max(img.size)
                    img = img.resize((int(img.size[0]*r), int(img.size[1]*r)), Image.Resampling.LANCZOS)
                if self.use_grayscale:
                    img = img.convert("L").convert("RGB")
                out = io.BytesIO()
                img.save(out, format="JPEG", quality=self.compression_quality, optimize=True)
                return out.getvalue()
        except Exception as e:
            log.error(f"optimize_image_bytes: {e}")
            return b

    def _crop_receipt(self, img_bgr: "np.ndarray") -> "np.ndarray":
        if not OPENCV:
            return img_bgr
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, binr = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        binr = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
        cnts,_ = cv2.findContours(binr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > 1000]
        if not cnts: return img_bgr
        x,y,w,h = cv2.boundingRect(max(cnts,key=cv2.contourArea))
        pad=50; x=max(0,x-pad); y=max(0,y-pad); w=min(img_bgr.shape[1]-x,w+2*pad); h=min(img_bgr.shape[0]-y,h+2*pad)
        cropped = img_bgr[y:y+h, x:x+w]
        log.info(f"Cropped to {cropped.shape[1]}x{cropped.shape[0]}")
        return cropped

    # ----- PDF (bytes) -----
    def _analyze_pdf_bytes(self, b: bytes) -> Dict:
        """Decide text-based vs image-based using PyPDF2; fall back to image if analysis fails."""
        if not PYPDF2:
            return {"type":"image","pages":1,"note":"PyPDF2 not available"}
        try:
            r = PyPDF2.PdfReader(io.BytesIO(b))
            if not r.pages: return {"type":"empty","pages":0}
            txt = (r.pages[0].extract_text() or "")
            if txt.strip() and len(txt) > 30:  # a bit lenient
                return {"type":"text","pages":len(r.pages),"text_length":len(txt)}
            return {"type":"image","pages":len(r.pages),"text_length":len(txt)}
        except Exception as e:
            log.warning(f"analyze_pdf_bytes: {e}")
            return {"type":"image","pages":1,"error":str(e)}

    def _pdf_firstpage_to_png_bytes(self, b: bytes) -> Optional[bytes]:
        """
        Render first page to PNG bytes using PyMuPDF only.
        Cropping is optional (OpenCV). No PIL dependency required for baseline path.
        """
        if not PYMUPDF:
            log.error("PyMuPDF not installed; cannot render PDF to image.")
            return None
        try:
            doc = fitz.open(stream=b, filetype="pdf")
            if len(doc) == 0: return None
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(2.0,2.0))
            png = pix.tobytes("png")
            if OPENCV:
                nparr = np.frombuffer(png, np.uint8)
                bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                bgr = self._crop_receipt(bgr)
                # re-encode cropped image as PNG
                ok, enc = cv2.imencode(".png", bgr)
                if ok:
                    return enc.tobytes()
                return png
            return png
        except Exception as e:
            log.error(f"pdf->image failed: {e}")
            return None

    # ----- OCR paths -----
    def _traditional_ocr(self, img_bytes: bytes) -> Dict:
        if not TRAD_OCR:
            return {"error":"Traditional OCR not available. Install tesseract or disable hybrid OCR."}
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                if img.mode != "RGB": img = img.convert("RGB")
                if max(img.size) > 1200:
                    r = 1200 / max(img.size)
                    img = img.resize((int(img.size[0]*r), int(img.size[1]*r)), Image.Resampling.LANCZOS)
                text = pytesseract.image_to_string(img)
            if not text.strip(): return {"error":"No text extracted","method":"traditional_ocr"}
            return {"raw_text":text, "parsed_data": self._parse_ocr_text(text),
                    "extraction_method":"traditional_ocr",
                    "confidence":"high" if len(text.strip())>50 else "low"}
        except Exception as e:
            log.error(f"traditional_ocr: {e}")
            return {"error":f"Traditional OCR failed: {e}", "method":"traditional_ocr"}

    def _vision_extract(self, img_bytes: bytes) -> Dict:
        """Call Ollama vision with raw bytes (already normalized)."""
        try:
            client = _ollama_client(ensure=self.model)
            resp = client.chat(
                model=self.model,
                messages=[{"role":"user","content": self._prompt(),"images":[img_bytes]}]
            )
            content = resp.get("message", {}).get("content", "")
            if not content: return {"error":"No response content from vision model"}
            return {"raw_response": content, "parsed_data": self._parse_chat_response(content),
                    "extraction_method":"vision_model","model_used": self.model}
        except Exception as e:
            log.error(f"vision_extract: {e}")
            return {"error": f"Ollama error: {e}"}

    # ----- Parsing -----
    def _parse_ocr_text(self, text: str) -> dict:
        """
        Robust parser for Vision OCR AND text-based PDFs (PyPDF2).
        Key fixes:
        - Do NOT treat 'Total' in the column header as summary start (require a money amount).
        - Multi-layout items: single-line, 2/3/4-line, and a token-based fallback.
        - Reconcile Subtotal/Tax/Total from parsed items.
        """
        import re

        # ---------- helpers ----------
        MONEY = r"\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})"
        MONEY_RE = re.compile(MONEY)
        INT_RE = re.compile(r"\d+")

        def normalize_ws(s: str) -> str:
            s = (s.replace("\u00a0", " ")
                .replace("\u2009", " ")
                .replace("\u202f", " ")
                .replace("\t", " ")
                .replace("|", " "))
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def is_money(s: str) -> bool:
            return bool(MONEY_RE.fullmatch(s.strip()))

        def first_money(s: str):
            m = MONEY_RE.search(s)
            return m.group(0) if m else None

        def two_moneys(s: str):
            ms = MONEY_RE.findall(s)
            return ms[:2] if len(ms) >= 2 else None

        def norm_money(s: str) -> str:
            s = s.strip().replace(" ", "")
            if s and not s.startswith("$") and is_money(s):
                return f"${s}"
            return s

        def money_to_float(s: str) -> float:
            return float(s.replace("$", "").replace(",", "").replace(" ", ""))

        def maybe_money_to_float(s: str):
            try:
                return money_to_float(s)
            except Exception:
                return None

        def find_qty(s: str):
            m = INT_RE.search(s.strip())
            return m.group(0) if m else None

        def looks_like_header(s: str) -> bool:
            ss = s.lower()
            return any(k in ss for k in ("description", "qty", "quantity", "unit price", "price", "total"))

        # Summary detection MUST include an amount either on the same line or the next non-empty line.
        def is_summary_line(lines, i) -> bool:
            ln = lines[i].lower()
            if re.search(r"\bsub\s*total\b|^subtotal\b|\b(tax|vat)\b|\bgrand\s+total\b", ln):
                # summary keywords beyond just 'total'
                return True
            if re.search(r"\btotal\b", ln):
                # require an amount on same line or next non-empty line
                if MONEY_RE.search(lines[i]):
                    return True
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and MONEY_RE.search(lines[j]):
                    return True
            return False

        # ---------- pre-process ----------
        raw_lines = text.splitlines()
        lines = [normalize_ws(ln) for ln in raw_lines if normalize_ws(ln)]
        low = [ln.lower() for ln in lines]

        # ---------- company ----------
        company = ""
        for ln in lines[:8]:
            if ln.startswith("---"):  # page marker
                continue
            if ":" in ln:             # meta like "Phone:", "Guest Name:"
                continue
            if looks_like_header(ln) or re.search(r"\b(receipt|invoice|statement)\b", ln, re.I):
                continue
            if MONEY_RE.search(ln):
                continue
            company = ln
            break

        # ---------- date ----------
        date_val = None
        for ln in lines:
            m = re.search(r"(check-?in|check in|\bdate\b)[:\s]+(.+)$", ln, re.I)
            if m:
                m2 = re.search(r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})|(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", m.group(2))
                if m2:
                    date_val = m2.group().strip()
                    break
        if not date_val:
            for ln in lines:
                m = re.search(r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})|(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", ln)
                if m:
                    date_val = m.group().strip()
                    break

        # ---------- locate table header (if present) ----------
        header_idx = None
        for i, ln in enumerate(low):
            if ("description" in ln) and ("total" in ln):
                header_idx = i
                break

        # ---------- summary detectors ----------
        SUBTOTAL_LINE = re.compile(rf"\bsub\s*total\b\s*:?\s*(?P<amt>{MONEY})\b", re.I)
        TAX_LINE      = re.compile(rf"\b(?:tax|vat)\b.*?:?\s*(?P<amt>{MONEY})\b", re.I)
        TOTAL_LINE    = re.compile(rf"\b(?:grand\s+)?total\b\s*:?\s*(?P<amt>{MONEY})\b", re.I)

        SINGLE_ITEM = re.compile(
            rf"""^
                (?P<desc>.+?)\s+
                (?P<qty>\d+)\s+
                (?P<unit>{MONEY})\s+
                (?P<total>{MONEY})\s*$
            """,
            re.VERBOSE | re.IGNORECASE
        )

        items = []
        subtotal_val = tax_val = total_val = None

        # determine item scan bounds: start after header; end at real summary (with amounts)
        start_i = (header_idx + 1) if header_idx is not None else 0

        end_i = len(lines)
        for k in range(start_i, len(lines)):
            if is_summary_line(lines, k):
                end_i = k
                break

        i = start_i
        while i < end_i:
            ln = lines[i]

            # Grab summary if it sneaks in early (rare)
            if subtotal_val is None:
                m = SUBTOTAL_LINE.search(ln)
                if m: subtotal_val = norm_money(m.group("amt"))
            if tax_val is None:
                m = TAX_LINE.search(ln)
                if m: tax_val = norm_money(m.group("amt"))
            if total_val is None:
                m = TOTAL_LINE.search(ln)
                if m: total_val = norm_money(m.group("amt"))

            # Skip clearly non-item lines
            if looks_like_header(ln) or (":" in ln and not ln.lower().startswith("total")):
                i += 1
                continue

            # A) single line
            m = SINGLE_ITEM.match(ln)
            if m:
                items.append({
                    "description": m.group("desc").strip(),
                    "quantity": m.group("qty").strip(),
                    "unit_price": norm_money(m.group("unit")),
                    "total": norm_money(m.group("total")),
                    "category": ""
                })
                i += 1
                continue

            # B) two-line: desc / (qty unit total)
            if i + 1 < end_i:
                l1 = lines[i + 1]
                qty = find_qty(l1)
                twom = two_moneys(l1)
                if qty and twom and not looks_like_header(ln) and not is_money(ln):
                    items.append({
                        "description": ln.strip(),
                        "quantity": qty,
                        "unit_price": norm_money(twom[0]),
                        "total": norm_money(twom[1]),
                        "category": ""
                    })
                    i += 2
                    continue

            # C) three-line: desc / qty / (unit total)
            if i + 2 < end_i:
                l1, l2 = lines[i + 1], lines[i + 2]
                qty = find_qty(l1)
                twom = two_moneys(l2)
                if qty and twom and not looks_like_header(ln) and not is_money(ln):
                    items.append({
                        "description": ln.strip(),
                        "quantity": qty,
                        "unit_price": norm_money(twom[0]),
                        "total": norm_money(twom[1]),
                        "category": ""
                    })
                    i += 3
                    continue

            # D) four-line: desc / qty / unit / total
            if i + 3 < end_i:
                l1, l2, l3 = lines[i + 1], lines[i + 2], lines[i + 3]
                qty = find_qty(l1)
                unit = first_money(l2)
                tot  = first_money(l3)
                if qty and unit and tot and not looks_like_header(ln) and not is_money(ln):
                    items.append({
                        "description": ln.strip(),
                        "quantity": qty,
                        "unit_price": norm_money(unit),
                        "total": norm_money(tot),
                        "category": ""
                    })
                    i += 4
                    continue

            # E) token-based fallback (very common for text-PDF):
            #    use last two money tokens and the last integer before them as qty
            tokens = ln.split()
            money_positions = [(idx, t) for idx, t in enumerate(tokens) if is_money(t)]
            if len(money_positions) >= 2:
                # rightmost two monies -> unit, total (or total, unit; try to detect by magnitude)
                idx2, m2 = money_positions[-1]
                idx1, m1 = money_positions[-2]
                # qty = last integer token before first money
                qty_idx = None
                for qidx in range(idx1 - 1, -1, -1):
                    if INT_RE.fullmatch(tokens[qidx]):
                        qty_idx = qidx
                        break
                if qty_idx is not None and qty_idx > 0:
                    desc = " ".join(tokens[:qty_idx]).strip()
                    unit, tot = m1, m2
                    # If unit looks bigger than total, swap (OCR glitches)
                    try:
                        if money_to_float(unit) > money_to_float(tot):
                            unit, tot = tot, unit
                    except Exception:
                        pass
                    if desc and not looks_like_header(desc) and not is_money(desc):
                        items.append({
                            "description": desc,
                            "quantity": tokens[qty_idx],
                            "unit_price": norm_money(unit),
                            "total": norm_money(tot),
                            "category": ""
                        })
                        i += 1
                        continue

            # nothing matched
            i += 1

        # Collect summary AFTER items region as well (safer)
        for k in range(end_i, len(lines)):
            if subtotal_val is None:
                m = SUBTOTAL_LINE.search(lines[k]);  subtotal_val = norm_money(m.group("amt")) if m else subtotal_val
            if tax_val is None:
                m = TAX_LINE.search(lines[k]);       tax_val = norm_money(m.group("amt")) if m else tax_val
            if total_val is None:
                m = TOTAL_LINE.search(lines[k]);     total_val = norm_money(m.group("amt")) if m else total_val

        # ---------- reconciliation ----------
        computed_subtotal = None
        if items:
            totals = [maybe_money_to_float(it["total"]) for it in items]
            totals = [t for t in totals if t is not None]
            if totals:
                computed_subtotal = round(sum(totals) + 1e-9, 2)

        def parse_money_safe(s: str | None) -> float | None:
            if not s: return None
            try: return money_to_float(s)
            except Exception: return None

        if computed_subtotal is not None:
            sub_f = parse_money_safe(subtotal_val)
            if (sub_f is None) or (abs(sub_f - computed_subtotal) > 0.01):
                subtotal_val = f"${computed_subtotal:.2f}"

        tax_f = parse_money_safe(tax_val)
        sub_f = parse_money_safe(subtotal_val)
        if tax_f is not None and sub_f is not None:
            total_val = f"${sub_f + tax_f:.2f}"

        if total_val is None:
            monies = [money_to_float(m.group()) for m in MONEY_RE.finditer("\n".join(lines))]
            if monies:
                total_val = f"${max(monies):.2f}"

        # ---------- build ----------
        out: dict = {}
        if company: out["Company Name"] = company
        if date_val: out["Date"] = date_val
        if items: out["Items"] = items
        if subtotal_val: out["Subtotal"] = subtotal_val
        if tax_val: out["Sales Tax"] = tax_val
        if total_val: out["Total"] = total_val
        out["raw_text"] = text
        return out




    def _parse_chat_response(self, content: str) -> Dict:
        import re
        m=re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                j=json.loads(m.group())
                cleaned={k:v for k,v in j.items() if str(v).strip() and str(v).lower() not in {"n/a","unknown","not found"}}
                if cleaned: return cleaned
            except json.JSONDecodeError:
                pass
        parsed={}
        pats={"company_name":r"(?:store|company|business)[:\s]+([^\n]+)",
              "total":r"(?:total|amount)[:\s]*\$?([\d,]+\.?\d*)",
              "date":r"(?:date)[:\s]+([^\n]+)"}
        for k,p in pats.items():
            mm=re.search(p, content, re.I)
            if mm:
                v=mm.group(1).strip()
                if v and v.lower() not in {"n/a","unknown","not found"}: parsed[k]=v
        parsed["raw_response"]=content
        return parsed

    # ----- Public API -----
    def extract_receipt_data_from_bytes(self, data, source_name: str = "upload") -> Dict:
        """
        Pass raw bytes (preferred), or a base64 / data-URL string.
        Auto-detects PDF vs image; uses text extraction for text-PDFs,
        and vision for images/image-PDFs. Hybrid OCR tried first on images if enabled.
        """
        t0 = time.time()
        try:
            payload = _normalize_to_bytes(data) if not isinstance(data, (bytes, bytearray)) else bytes(data)
            kind = _detect_payload_type(payload)

            if kind == "pdf":
                log.info(f"Processing PDF bytes for {source_name}")
                analysis = self._analyze_pdf_bytes(payload)
                log.info(f"PDF analysis: {analysis}")

                if analysis.get("type") == "text":
                    # Text-based PDF: direct text extraction (no vision)
                    result = self._extract_text_from_pdf_bytes(payload)
                    result["processing_time"] = time.time() - t0
                    return result

                # Image-based PDF: render to PNG, then do image path
                img = self._pdf_firstpage_to_png_bytes(payload)
                if not img:
                    return {"error":"PDF seems image-based but conversion failed", "processing_time": time.time() - t0}
                res = self._process_image_bytes(img)
                res["pdf_processing"] = True
                return res

            elif kind == "image":
                return self._process_image_bytes(payload)

            else:
                return {"error":"Unsupported payload type (not a valid PDF or image)", "processing_time": time.time() - t0}

        except Exception as e:
            log.error(f"buffer processing error for {source_name}: {e}")
            return {"error":str(e), "source":source_name, "timestamp":datetime.now().isoformat(),
                    "processing_time": time.time() - t0}

    def _extract_text_from_pdf_bytes(self, b: bytes) -> Dict:
        if not PYPDF2:
            return {"error":"PDF text extraction unavailable (PyPDF2 not installed)"}
        try:
            r=PyPDF2.PdfReader(io.BytesIO(b)); text=""
            for i,p in enumerate(r.pages,1):
                t=p.extract_text() or ""
                if t: text += f"\n--- Page {i} ---\n{t}"
            if not text.strip(): return {"error":"No text content found in PDF"}
            parsed = self._parse_ocr_text(text)
            return {
                "raw_text": text,
                "parsed_data": parsed,
                "extraction_method": "pdf_text_extraction",
                "confidence": "high" if len(text.strip()) > 100 else "low"
            }
        except Exception as e:
            log.error(f"pdf text extraction failed: {e}")
            return {"error":f"PDF text extraction failed: {e}"}

    def _process_image_bytes(self, img_bytes: bytes) -> Dict:
        t0 = time.time()
        # Try traditional OCR first (fast) if available
        if self.use_hybrid_ocr:
            log.info("Attempting traditional OCR first...")
            ocr = self._traditional_ocr(img_bytes)
            if "error" not in ocr and ocr.get("confidence") == "high":
                ocr["processing_time"] = time.time() - t0
                self.processing_stats["traditional_ocr_success"] += 1
                log.info(f"Traditional OCR successful in {ocr['processing_time']:.2f}s")
                return ocr
        # Use vision
        log.info("Using vision model for extraction...")
        self.processing_stats["vision_model_used"] += 1
        optimized = self._optimize_image_bytes(img_bytes)
        res = self._vision_extract(optimized)
        res["processing_time"] = time.time() - t0
        log.info(f"Vision model done in {res['processing_time']:.2f}s")
        return res

    # ----- Comparison & Save (kept for parity; not used in text-PDF path) -----
    def save_results(self, results: Dict, source_name: str = "upload") -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model.split(":")[0] if ":" in self.model else self.model
        safe = (source_name.replace("/","_").replace("\\","_").replace(" ","_")[:40] or "upload")
        out = self.output_dir / f"{safe}_{model_name}_optimized_{ts}.json"
        payload = {
            "metadata": {
                "source_name": source_name, "model_used": self.model,
                "extraction_timestamp": ts, "processing_time": results.get("processing_time",0),
                "optimization_settings": {
                    "max_image_size": self.max_image_size,
                    "use_grayscale": self.use_grayscale,
                    "use_hybrid_ocr": self.use_hybrid_ocr,
                    "compression_quality": self.compression_quality
                }
            },
            "results": results
        }
        json.dump(payload, open(out,"w",encoding="utf-8"), indent=2, ensure_ascii=False)
        log.info(f"Results saved to {out}")
        return out

# ---------- CLI ----------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Receipt OCR Scanner (buffer-based)")
    p.add_argument("--model", default="llava:7b")
    p.add_argument("--input", required=True, help="Path to image or PDF")
    p.add_argument("--output-dir", default="./Reports")
    p.add_argument("--max-image-size", type=int, default=800)
    p.add_argument("--use-grayscale", action="store_true", default=True)
    p.add_argument("--no-grayscale", dest="use_grayscale", action="store_false")
    p.add_argument("--hybrid-ocr", action="store_true", default=True)
    p.add_argument("--no-hybrid-ocr", dest="use_hybrid_ocr", action="store_false")
    p.add_argument("--compression-quality", type=int, default=85)
    p.add_argument("--verbose","-v",action="store_true")
    a = p.parse_args()
    if a.verbose: logging.getLogger().setLevel(logging.DEBUG)

    scanner = OptimizedReceiptOCRScanner(
        output_dir=a.output_dir, model=a.model,
        max_image_size=a.max_image_size, use_grayscale=a.use_grayscale,
        use_hybrid_ocr=a.use_hybrid_ocr, compression_quality=a.compression_quality
    )

    path = Path(a.input)
    if not path.exists() or not path.is_file():
        log.error(f"Input file not found: {path}"); sys.exit(1)
    data = path.read_bytes()
    res = scanner.extract_receipt_data_from_bytes(data, source_name=path.name)
    out = scanner.save_results(res, source_name=path.name)
    print(f"Saved results: {out}")

if __name__ == "__main__":
    main()
