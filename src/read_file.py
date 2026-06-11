import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.tools import tool

MAX_CHARS = 10000


def _read_pdf(file_path: str) -> str:
    """Read PDF with pymupdf. Falls back to OCR for scanned/image-based PDFs."""
    try:
        import fitz  # pymupdf
    except ImportError:
        return _read_pdf_pypdf(file_path)

    try:
        doc = fitz.open(file_path)
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text:
                pages.append(text.strip())
            else:
                # page has no text layer — try OCR
                ocr_text = _ocr_page(page)
                if ocr_text:
                    pages.append(ocr_text.strip())
        doc.close()
        content = "\n\n".join(pages)
        if not content.strip():
            return "[PDF 无法提取文字] 该文件可能是扫描件，OCR 也未识别出文字。"
        return _trim(content)
    except Exception:
        return _read_pdf_pypdf(file_path)


_easyocr_reader = None


def _get_easyocr_reader():
    """Lazy-init easyocr reader (Chinese + English)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
    return _easyocr_reader


def _ocr_page(page) -> str | None:
    """Try OCR on a single PDF page using easyocr."""
    try:
        import tempfile, os, numpy as np
        reader = _get_easyocr_reader()
        pix = page.get_pixmap(dpi=200)
        # Convert pymupdf pixmap to numpy array for easyocr
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            img_array = img_array[:, :, :3]  # drop alpha
        results = reader.readtext(img_array, detail=0)
        return "\n".join(results) if results else None
    except Exception:
        return None


def _read_pdf_pypdf(file_path: str) -> str:
    """Fallback PDF reader using PyPDFLoader."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    pages = [d.page_content.strip() for d in docs if d.page_content.strip()]
    return _trim("\n\n".join(pages))


def _trim(content: str) -> str:
    """Clean up whitespace and truncate to MAX_CHARS."""
    # collapse 3+ newlines into 2
    import re
    content = re.sub(r"\n{3,}", "\n\n", content)
    # collapse multiple spaces (but not newlines)
    content = re.sub(r"[ \t]{3,}", "  ", content)
    if len(content) > MAX_CHARS:
        content = content[:MAX_CHARS] + "... [truncated]"
    return content


def read_file(file_path):
    """A tool to read files of different types, including txt, pdf, md, csv and docx. Returns file content (first 10000 characters)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    extend = file_path.split(".")[-1].lower()

    if extend == 'pdf':
        try:
            return _read_pdf(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read file {file_path}: {e}")

    loaders = {
        'txt': TextLoader(file_path, encoding='utf-8'),
        'md': UnstructuredMarkdownLoader(file_path),
        'csv': CSVLoader(file_path),
        'docx': UnstructuredWordDocumentLoader(file_path),
    }

    if extend not in loaders:
        raise ValueError(f"Unsupported file type: {extend}")

    try:
        docs = loaders[extend].load()
        content = "\n\n".join(d.page_content for d in docs) if docs else ""
        return _trim(content)
    except Exception as e:
        raise RuntimeError(f"Failed to read file {file_path}: {e}")