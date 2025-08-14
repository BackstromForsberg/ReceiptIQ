#!/usr/bin/env python3
"""
Optimized Receipt OCR Scanner (buffer-based)
- Accepts image/PDF bytes (array buffers) instead of file paths
- Robust Ollama integration: lazy client, DNS/HTTP waits, auto-pull model
"""

import sys, os, argparse, json, time, io, socket, logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# ---------- Optional deps (OCR / imaging) ----------
try:
    import pytesseract
    from PIL import Image
    TRADITIONAL_OCR_AVAILABLE = True
except Exception:
    TRADITIONAL_OCR_AVAILABLE = False
    print("Warning: pytesseract not available. Install: pip install pytesseract pillow")

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False
    print("Warning: opencv-python not available. Install: pip install opencv-python")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except Exception:
    PYPDF2_AVAILABLE = False

# ---------- Ollama ----------
try:
    import requests
except Exception:
    print("Error: requests not found. Install: pip install requests")
    sys.exit(1)

try:
    import ollama
except Exception:
    print("Error: ollama not found. Install: pip install ollama")
    sys.exit(1)

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("receipt_ocr_optimized.log")]
)
logger = logging.getLogger(__name__)

# ---------- Helpers ----------
def _looks_like_pdf(b: bytes) -> bool:
    return len(b) >= 4 and b[:4] == b"%PDF"

# ---------- Robust Ollama layer ----------
def _env_ollama_host() -> str:
    raw = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
    if raw.startswith("//"): raw = "http:" + raw
    if not raw.startswith(("http://", "https://")): raw = "http://" + raw
    return raw.rstrip("/")

def _split_host_port(url: str) -> tuple[str, int]:
    h = url.split("://", 1)[1]
    if "/" in h: h = h.split("/", 1)[0]
    host, _, port = h.partition(":")
    return host, int(port or "11434")

def _wait_dns(host: str, timeout_s: float = 30.0):
    deadline = time.time() + timeout_s
    while True:
        try:
            socket.getaddrinfo(host, None)
            return
        except socket.gaierror:
            if time.time() >= deadline:
                raise RuntimeError(f"DNS resolve failed for '{host}' after {timeout_s:.0f}s")
            time.sleep(0.5)

def _wait_http(url: str, timeout_s: float = 120.0):
    endpoint = f"{url}/api/tags"
    deadline = time.time() + timeout_s
    while True:
        try:
            r = requests.get(endpoint, timeout=2)
            if r.ok:
                return
        except Exception:
            pass
        if time.time() >= deadline:
            raise RuntimeError(f"Ollama not responding at {endpoint} after {timeout_s:.0f}s")
        time.sleep(0.5)

def _ensure_model(url: str, model: str, timeout_s: float = 900.0):
    try:
        t = requests.get(f"{url}/api/tags", timeout=5)
        if t.ok:
            models = [m.get("name") for m in t.json().get("models", [])]
            if model in models:
                return
    except Exception:
        pass
    logger.info(f"Pulling model '{model}' ...")
    s = requests.Session()
    resp = s.post(f"{url}/api/pull", json={"name": model}, stream=True, timeout=15)
    resp.raise_for_status()
    start = time.time()
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if "error" in (line or "").lower():
            raise RuntimeError(f"ollama pull error: {line}")
        if '"done":true' in line or '"status":"success"' in line or '"status":"downloaded"' in line:
            break
        if time.time() - start > timeout_s:
            raise RuntimeError(f"Timed out pulling '{model}' after {timeout_s:.0f}s")
    logger.info(f"Model '{model}' ready.")

_OLLAMA_CLIENT = None
_OLLAMA_HOST = None

def _ollama_client(ensure_model: Optional[str] = None):
    """Create/reuse a client bound to OLLAMA_HOST; wait DNS/HTTP; optionally pull model."""
    global _OLLAMA_CLIENT, _OLLAMA_HOST
    if _OLLAMA_CLIENT is not None:
        return _OLLAMA_CLIENT
    url = _env_ollama_host()
    host, _ = _split_host_port(url)
    logger.info(f"Ollama target: {url}")
    _wait_dns(host, 30.0)
    _wait_http(url, 120.0)
    if ensure_model:
        _ensure_model(url, ensure_model, 900.0)
    _OLLAMA_HOST = url
    _OLLAMA_CLIENT = ollama.Client(host=url)
    return _OLLAMA_CLIENT

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
        self.use_hybrid_ocr = use_hybrid_ocr and TRADITIONAL_OCR_AVAILABLE
        self.compression_quality = compression_quality
        self.template = self._load_template()
        self.processing_stats = {"traditional_ocr_success": 0, "vision_model_used": 0}
        logger.info(f"Initialized OptimizedReceiptOCRScanner model={model} max={max_image_size} gray={use_grayscale} hybrid={self.use_hybrid_ocr}")

    # ----- Template / Prompt -----
    def _load_template(self) -> Dict:
        p = Path("Template/llama3.2-vision_template.json")
        if p.exists():
            try:
                return json.load(open(p, "r", encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Could not load template: {e}")
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
        if not TRADITIONAL_OCR_AVAILABLE:
            return b
        try:
            with Image.open(io.BytesIO(b)) as img:
                if img.mode != "RGB": img = img.convert("RGB")
                if max(img.size) > self.max_image_size:
                    r = self.max_image_size / max(img.size)
                    img = img.resize((int(img.size[0]*r), int(img.size[1]*r)), Image.Resampling.LANCZOS)
                if self.use_grayscale: img = img.convert("L").convert("RGB")
                out = io.BytesIO(); img.save(out, format="JPEG", quality=self.compression_quality, optimize=True)
                return out.getvalue()
        except Exception as e:
            logger.error(f"optimize_image_bytes: {e}")
            return b

    def _crop_receipt(self, img_bgr: "np.ndarray") -> "np.ndarray":
        if not OPENCV_AVAILABLE: return img_bgr
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, binr = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        binr = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
        cnts,_ = cv2.findContours(binr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > 1000]
        if not cnts: return img_bgr
        x,y,w,h = cv2.boundingRect(max(cnts,key=cv2.contourArea))
        pad=50; x=max(0,x-pad); y=max(0,y-pad); w=min(img_bgr.shape[1]-x,w+2*pad); h=min(img_bgr.shape[0]-y,h+2*pad)
        cropped = img_bgr[y:y+h, x:x+w]
        logger.info(f"Cropped to {cropped.shape[1]}x{cropped.shape[0]}")
        return cropped

    # ----- PDF (bytes) -----
    def _analyze_pdf_bytes(self, b: bytes) -> Dict:
        if not PYPDF2_AVAILABLE:
            return {"type":"image","pages":1,"note":"PyPDF2 not available"}
        try:
            r = PyPDF2.PdfReader(io.BytesIO(b))
            if not r.pages: return {"type":"empty","pages":0}
            txt = (r.pages[0].extract_text() or "")
            if txt.strip() and len(txt)>50:
                return {"type":"text","pages":len(r.pages),"text_length":len(txt),
                        "sample_text": (txt[:200]+"...") if len(txt)>200 else txt}
            return {"type":"image","pages":len(r.pages),"text_length":len(txt)}
        except Exception as e:
            logger.warning(f"analyze_pdf_bytes: {e}")
            return {"type":"image","pages":1,"error":str(e)}

    def _pdf_firstpage_to_png_bytes(self, b: bytes) -> Optional[bytes]:
        if not (PYMUPDF_AVAILABLE and OPENCV_AVAILABLE and TRADITIONAL_OCR_AVAILABLE):
            return None
        try:
            doc = fitz.open(stream=b, filetype="pdf")
            if len(doc)==0: return None
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(2.0,2.0))
            nparr = np.frombuffer(pix.tobytes("png"), np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            bgr = self._crop_receipt(bgr)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            if pil.mode != "RGB": pil = pil.convert("RGB")
            if max(pil.size) > self.max_image_size:
                r = self.max_image_size / max(pil.size)
                pil = pil.resize((int(pil.size[0]*r), int(pil.size[1]*r)), Image.Resampling.LANCZOS)
            if self.use_grayscale: pil = pil.convert("L").convert("RGB")
            out = io.BytesIO(); pil.save(out, format="PNG", optimize=True)
            return out.getvalue()
        except Exception as e:
            logger.error(f"pdf->image failed: {e}")
            return None

    # ----- OCR paths -----
    def _traditional_ocr(self, img_bytes: bytes) -> Dict:
        if not TRADITIONAL_OCR_AVAILABLE:
            return {"error":"Traditional OCR not available"}
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
            logger.error(f"traditional_ocr: {e}")
            return {"error":f"Traditional OCR failed: {e}", "method":"traditional_ocr"}

    def _vision_extract(self, img_bytes: bytes) -> Dict:
        try:
            client = _ollama_client(ensure_model=self.model)
            resp = client.chat(
                model=self.model,
                messages=[{"role":"user","content": self._prompt(),"images":[img_bytes]}]
            )
            content = resp.get("message", {}).get("content", "")
            if not content: return {"error":"No response content from vision model"}
            return {"raw_response": content, "parsed_data": self._parse_chat_response(content),
                    "extraction_method":"vision_model","model_used": self.model}
        except Exception as e:
            logger.error(f"vision_extract: {e}")
            return {"error": f"Ollama error: {e}"}

    # ----- Parsing -----
    def _parse_ocr_text(self, text: str) -> dict:
        import re
        MONEY = r"\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})"
        def is_money(s): return bool(re.fullmatch(MONEY, s.strip()))
        def norm_money(s): s=s.strip(); return s if s.startswith("$") else (f"${s}" if is_money(s) else s)
        def to_float(m): return float(m.replace("$","").replace(",",""))
        def is_qty(s): return bool(re.fullmatch(r"\d+", s.strip()))
        def is_hdr(s): return s.strip().lower() in {"description","qty","quantity","unit price","price","total"}
        def next_amt(i, lines):
            j=i+1
            while j<len(lines) and not lines[j].strip(): j+=1
            return norm_money(lines[j]) if j<len(lines) and is_money(lines[j]) else None

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        low = [ln.lower() for ln in lines]

        company=""
        for ln in lines[:8]:
            if ln.startswith("---") or ":" in ln or is_hdr(ln) or re.search(r"receipt|invoice|statement", ln, re.I) or is_money(ln) or is_qty(ln):
                continue
            company = ln; break

        date_val=None
        for ln in lines:
            m=re.search(r"(check-?in|check in)\s*date[:\s]+(.+)$", ln, re.I)
            if m:
                m2=re.search(r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})|(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", m.group(2))
                if m2: date_val=m2.group(); break
        if not date_val:
            for ln in lines:
                m=re.search(r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})|(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", ln)
                if m: date_val=m.group(); break

        # header find
        header_idx=None
        for i in range(len(lines)):
            win={lines[i].lower()}
            if i+1<len(lines): win.add(lines[i+1].lower())
            if i+2<len(lines): win.add(lines[i+2].lower())
            if i+3<len(lines): win.add(lines[i+3].lower())
            if {"description","total"}.issubset({w.strip().lower() for w in win}):
                header_idx=i; break

        items=[]; items_end=None
        if header_idx is not None:
            j=header_idx
            while j<len(lines) and is_hdr(lines[j]): j+=1
            while j+3<len(lines):
                if re.search(r"^\s*(sub\s*total|subtotal|tax|vat|total|grand\s+total)\b", lines[j], re.I):
                    items_end=j; break
                desc, qty, unit, tot = lines[j], lines[j+1], lines[j+2], lines[j+3]
                desc_ok=(not is_hdr(desc)) and (":" not in desc) and (not is_money(desc)) and (not is_qty(desc))
                if desc_ok and is_qty(qty) and is_money(unit) and is_money(tot):
                    items.append({"description":desc,"quantity":qty,"unit_price":norm_money(unit),"total":norm_money(tot),"category":""})
                    j+=4
                else:
                    j+=1
            if items_end is None: items_end=j
        start_summary = items_end if items_end is not None else 0

        subtotal_val=tax_val=total_val=None
        for i in range(start_summary, len(lines)):
            ln, lw = lines[i], low[i]
            if subtotal_val is None and re.search(r"\bsub\s*total\b|^subtotal\b", lw):
                m=re.search(MONEY+r"\s*$", ln); subtotal_val=norm_money(m.group()) if m else next_amt(i, lines); continue
            if tax_val is None and re.search(r"\b(tax|vat)\b", lw) and not re.search(r"\bno\s*tax\b", lw):
                m=re.search(MONEY+r"\s*$", ln); tax_val=norm_money(m.group()) if m else next_amt(i, lines); continue
            if re.fullmatch(r"(?i)(?:grand\s+)?total[:\s]*.*", ln):
                m=re.search(MONEY+r"\s*$", ln); total_val=norm_money(m.group()) if m else next_amt(i, lines); continue

        if total_val is None and subtotal_val and tax_val:
            total_val = f"${to_float(subtotal_val)+to_float(tax_val):.2f}"
        if total_val and subtotal_val and tax_val:
            if abs(to_float(total_val)-to_float(subtotal_val))<0.005:
                total_val = f"${to_float(subtotal_val)+to_float(tax_val):.2f}"
        if total_val is None:
            import re as _re
            monies = _re.findall(MONEY, "\n".join(lines[start_summary:]))
            if monies:
                def _tf(s): return float(s.replace("$","").replace(",",""))
                total_val = norm_money(max(monies, key=_tf))

        out={}
        if company: out["Company Name"]=company
        if date_val: out["Date"]=date_val
        if items: out["Items"]=items
        if subtotal_val: out["Subtotal"]=subtotal_val
        if tax_val: out["Sales Tax"]=tax_val
        if total_val: out["Total"]=total_val
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
    def extract_receipt_data_from_bytes(self, data: bytes, source_name: str = "upload") -> Dict:
        start=time.time()
        try:
            if _looks_like_pdf(data):
                logger.info(f"Processing PDF bytes for {source_name}")
                analysis=self._analyze_pdf_bytes(data); logger.info(f"PDF analysis: {analysis}")
                if analysis.get("type")=="text":
                    text_result=self._extract_text_from_pdf_bytes(data)
                    if "error" not in text_result:
                        img=self._pdf_firstpage_to_png_bytes(data)
                        if img:
                            image_result=self._process_image_bytes(img)
                            cmp=self._compare_text_vs_image_results(text_result, image_result)
                            cmp["processing_time"]=time.time()-start; cmp["pdf_processing"]=True; cmp["pdf_analysis"]=analysis
                            return cmp
                        text_result["processing_time"]=time.time()-start; return text_result
                    else:
                        img=self._pdf_firstpage_to_png_bytes(data)
                        if not img: return {"error":"Failed PDF text + image conversion"}
                        return self._process_image_bytes(img)
                else:
                    img=self._pdf_firstpage_to_png_bytes(data)
                    if not img: return {"error":"PDF seems image-based but conversion failed"}
                    res=self._process_image_bytes(img); res["pdf_processing"]=True; return res
            else:
                return self._process_image_bytes(data)
        except Exception as e:
            logger.error(f"buffer processing error for {source_name}: {e}")
            return {"error":str(e),"source":source_name,"timestamp":datetime.now().isoformat(),
                    "processing_time": time.time()-start}

    def _extract_text_from_pdf_bytes(self, b: bytes) -> Dict:
        if not PYPDF2_AVAILABLE:
            return {"error":"PDF text extraction unavailable (PyPDF2 not installed)"}
        try:
            r=PyPDF2.PdfReader(io.BytesIO(b)); text=""
            for i,p in enumerate(r.pages,1):
                t=p.extract_text() or ""
                if t: text += f"\n--- Page {i} ---\n{t}"
            if not text.strip(): return {"error":"No text content found in PDF"}
            return {"raw_text":text,"parsed_data":self._parse_ocr_text(text),
                    "extraction_method":"pdf_text_extraction",
                    "confidence":"high" if len(text.strip())>100 else "low"}
        except Exception as e:
            logger.error(f"pdf text extraction failed: {e}")
            return {"error":f"PDF text extraction failed: {e}"}

    def _process_image_bytes(self, img_bytes: bytes) -> Dict:
        start=time.time()
        if self.use_hybrid_ocr:
            logger.info("Attempting traditional OCR first...")
            ocr=self._traditional_ocr(img_bytes)
            if "error" not in ocr and ocr.get("confidence")=="high":
                ocr["processing_time"]=time.time()-start
                self.processing_stats["traditional_ocr_success"]+=1
                logger.info(f"Traditional OCR successful in {ocr['processing_time']:.2f}s")
                return ocr
        logger.info("Using vision model for extraction...")
        self.processing_stats["vision_model_used"]+=1
        optimized=self._optimize_image_bytes(img_bytes)
        res=self._vision_extract(optimized)
        res["processing_time"]=time.time()-start
        logger.info(f"Vision model done in {res['processing_time']:.2f}s")
        return res

    # ----- Comparison / Save -----
    def _compare_text_vs_image_results(self, text_result: Dict, image_result: Dict) -> Dict:
        text_fields=len([k for k,v in text_result.get("parsed_data",{}).items() if str(v).strip()])
        image_fields=len([k for k,v in image_result.get("parsed_data",{}).items() if str(v).strip()])
        text_score=2 if text_fields>image_fields else 0
        image_score=2 if image_fields>text_fields else 0
        tt=text_result.get("processing_time",0); it=image_result.get("processing_time",0)
        if tt<it: text_score+=1
        elif it<tt: image_score+=1
        conf={"high":3,"medium":2,"low":1}
        text_score+=conf.get(text_result.get("confidence","low"),1)
        image_score+=conf.get(image_result.get("confidence","low"),1)
        winner="text" if text_score>=image_score else "image"
        return {"winner":winner,"text_result":text_result,"image_result":image_result,
                "comparison_scores":{"text_score":text_score,"image_score":image_score,
                                     "text_fields":text_fields,"image_fields":image_fields,
                                     "text_time":tt,"image_time":it,
                                     "text_confidence":text_result.get("confidence","low"),
                                     "image_confidence":image_result.get("confidence","low")},
                "extraction_method":f"pdf_{winner}_based","pdf_processing":True}

    def save_results(self, results: Dict, source_name: str = "upload") -> Path:
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name=self.model.split(":")[0] if ":" in self.model else self.model
        safe=(source_name.replace("/","_").replace("\\","_").replace(" ","_")[:40] or "upload")
        out=self.output_dir / f"{safe}_{model_name}_optimized_{ts}.json"
        payload={"metadata":{"source_name":source_name,"model_used":self.model,"extraction_timestamp":ts,
                             "processing_time":results.get("processing_time",0),
                             "optimization_settings":{"max_image_size":self.max_image_size,
                                                      "use_grayscale":self.use_grayscale,
                                                      "use_hybrid_ocr":self.use_hybrid_ocr,
                                                      "compression_quality":self.compression_quality}},
                 "results":results}
        json.dump(payload, open(out,"w",encoding="utf-8"), indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {out}")
        return out

# ---------- CLI (debug/local) ----------
def main():
    parser = argparse.ArgumentParser(description="Optimized Receipt OCR Scanner (buffer-based)")
    parser.add_argument("--model", default="llava:7b", help="Vision model to use")
    parser.add_argument("--input", required=True, help="Path to a single file (image or PDF)")
    parser.add_argument("--output-dir", default="./Reports", help="Directory to save results")
    parser.add_argument("--max-image-size", type=int, default=800)
    parser.add_argument("--use-grayscale", action="store_true", default=True)
    parser.add_argument("--no-grayscale", dest="use_grayscale", action="store_false")
    parser.add_argument("--hybrid-ocr", action="store_true", default=True)
    parser.add_argument("--no-hybrid-ocr", dest="use_hybrid_ocr", action="store_false")
    parser.add_argument("--compression-quality", type=int, default=85)
    parser.add_argument("--verbose","-v",action="store_true")
    args = parser.parse_args()
    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)

    s = OptimizedReceiptOCRScanner(output_dir=args.output_dir, model=args.model,
                                   max_image_size=args.max_image_size,
                                   use_grayscale=args.use_grayscale,
                                   use_hybrid_ocr=args.use_hybrid_ocr,
                                   compression_quality=args.compression_quality)
    p = Path(args.input)
    if not p.exists() or not p.is_file():
        logger.error(f"Input file not found: {p}"); sys.exit(1)
    data = p.read_bytes()
    res = s.extract_receipt_data_from_bytes(data, source_name=p.name)
    out = s.save_results(res, source_name=p.name)
    print(f"Saved results: {out}")

if __name__ == "__main__":
    main()
