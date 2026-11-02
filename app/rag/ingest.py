"""Load PDF / DOCX / Markdown / text into section- and page-aware chunks."""

import io
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.rag.models import Chunk

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt"}

# (page, section, text) blocks before size-based splitting
Block = tuple[int | None, str | None, str]


class UnsupportedFileError(ValueError):
    pass


def _load_markdown(text: str) -> list[Block]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False
    )
    blocks: list[Block] = []
    for doc in splitter.split_text(text):
        meta = doc.metadata
        section = " > ".join(meta[k] for k in ("h2", "h3") if k in meta) or meta.get("h1")
        blocks.append((None, section, doc.page_content))
    return blocks


def _load_pdf(data: bytes) -> list[Block]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return [(i, None, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]


def _load_docx(data: bytes) -> list[Block]:
    import docx

    document = docx.Document(io.BytesIO(data))
    blocks: list[Block] = []
    section: str | None = None
    buf: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style is not None and para.style.name.lower().startswith("heading"):
            if buf:
                blocks.append((None, section, "\n".join(buf)))
                buf = []
            section = text
        buf.append(text)
    if buf:
        blocks.append((None, section, "\n".join(buf)))
    return blocks


def load_blocks(filename: str, data: bytes) -> list[Block]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileError(f"unsupported file type: {suffix or '(none)'}")
    if suffix == ".pdf":
        return _load_pdf(data)
    if suffix == ".docx":
        return _load_docx(data)
    text = data.decode("utf-8", errors="replace")
    return _load_markdown(text) if suffix in {".md", ".markdown"} else [(None, None, text)]


def chunk_document(filename: str, data: bytes, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    name = Path(filename).name
    chunks: list[Chunk] = []
    for page, section, text in load_blocks(filename, data):
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if len(piece) >= 20:  # drop heading-only / empty fragments
                chunks.append(Chunk(doc_name=name, content=piece, page=page, section=section))
    return chunks
