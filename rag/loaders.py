"""Document loading: turn PDF / DOCX / TXT / MD files into (text, metadata) pages."""
import io
import os
from typing import List, Tuple

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")

Page = Tuple[str, dict]   # (text, {"source": filename, "page": int})


def _clean(text: str) -> str:
    # normalise whitespace but keep paragraph breaks (the chunker splits on them)
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    out, blank = [], 0
    for ln in lines:
        blank = blank + 1 if not ln.strip() else 0
        if blank <= 1:
            out.append(ln)
    return "\n".join(out).strip()


def load_bytes(name: str, data: bytes) -> List[Page]:
    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = [(_clean(p.extract_text() or ""), {"source": name, "page": i + 1})
                 for i, p in enumerate(reader.pages)]
    elif ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        pages = [(_clean("\n\n".join(p.text for p in doc.paragraphs)), {"source": name, "page": 1})]
    else:
        pages = [(_clean(data.decode("utf-8", errors="replace")), {"source": name, "page": 1})]

    pages = [(t, m) for t, m in pages if t]
    if not pages:
        raise ValueError(f"No extractable text in '{name}' (scanned PDFs need OCR).")
    return pages


def load_path(path: str) -> List[Page]:
    with open(path, "rb") as f:
        return load_bytes(os.path.basename(path), f.read())
