import re
import io
from typing import List, Dict


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}")


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        raise RuntimeError(f"DOCX extraction failed: {e}")


def extract_text(file_bytes: bytes, file_name: str) -> str:
    ext = file_name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext == "docx":
        return extract_text_from_docx(file_bytes)
    else:
        return extract_text_from_txt(file_bytes)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    source: str = "unknown",
) -> List[Dict]:
    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append({
            "text": chunk_text_str,
            "source": source,
            "chunk_index": chunk_index,
            "word_count": len(chunk_words),
        })

        chunk_index += 1
        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks


def process_document(file_bytes: bytes, file_name: str) -> List[Dict]:
    raw_text = extract_text(file_bytes, file_name)
    if not raw_text.strip():
        raise ValueError(f"No text could be extracted from '{file_name}'.")
    return chunk_text(raw_text, source=file_name)
