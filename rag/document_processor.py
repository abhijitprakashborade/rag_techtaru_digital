"""
rag/document_processor.py
=========================
Handles all PDF ingestion: loading, text extraction, chunking,
SHA-256 deduplication, and metadata enrichment.

pdfplumber is the primary extractor; pypdf is a fallback for
files pdfplumber struggles with (some scanned-with-OCR or
non-standard-encoding PDFs).

Chunks are produced via LangChain's RecursiveCharacterTextSplitter.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMPTY_PDF_ERROR,
    LARGE_FILE_WARNING,
    PDF_LARGE_PAGE_COUNT,
    PDF_LARGE_SIZE_MB,
    PDF_MIN_TEXT_LENGTH,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────
@dataclass
class ProcessedPDF:
    """Result of a successful PDF processing operation.

    Attributes:
        filename: Original file name.
        sha256: SHA-256 hex digest of raw file bytes (for dedup).
        page_count: Total number of pages extracted.
        char_count: Total characters extracted across all pages.
        chunks: LangChain Document objects ready for embedding.
        is_large: True when the file exceeded size/page thresholds.
    """

    filename: str
    sha256: str
    page_count: int
    char_count: int
    chunks: list[Document] = field(default_factory=list)
    is_large: bool = False


@dataclass
class ProcessingError:
    """Represents a non-fatal processing failure.

    Attributes:
        filename: Original file name.
        message: Human-readable error description shown in the UI.
    """

    filename: str
    message: str


# ─────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────
def compute_sha256(file_bytes: bytes) -> str:
    """Compute the SHA-256 hex digest of raw file bytes.

    Args:
        file_bytes: Raw bytes of the uploaded file.

    Returns:
        64-character hexadecimal SHA-256 digest string.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def _extract_with_pdfplumber(
    file_bytes: bytes,
    filename: str,
) -> tuple[list[tuple[str, int]], int]:
    """Extract text with pdfplumber, page by page.

    Returns:
        (pages_data, total_pages) — per-page text and 1-indexed page numbers.

    Raises:
        ValueError: If the PDF cannot be opened at all.
    """
    import io

    pages_data: list[tuple[str, int]] = []
    total_pages: int = 0

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total_pages = len(pdf.pages)
            logger.info("Opened '%s' — %d pages found.", filename, total_pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    text: str | None = page.extract_text()
                    if text and text.strip():
                        pages_data.append((text.strip(), page_num))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Page %d in '%s' could not be extracted: %s",
                        page_num,
                        filename,
                        exc,
                    )
    except Exception as exc:
        logger.warning("pdfplumber failed to open '%s': %s", filename, exc)
        raise ValueError(f"Cannot open '{filename}': {exc}") from exc

    return pages_data, total_pages


def _extract_with_pypdf(
    file_bytes: bytes,
    filename: str,
) -> tuple[list[tuple[str, int]], int]:
    """Fallback extractor using pypdf.

    Handles PDFs pdfplumber rejects (non-standard encodings, some
    OCR'd documents). Returns the same shape as the pdfplumber path.
    """
    import io

    pages_data: list[tuple[str, int]] = []
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)
        logger.info("pypdf opened '%s' — %d pages.", filename, total_pages)
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
                if text.strip():
                    pages_data.append((text.strip(), page_num))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "pypdf could not extract page %d in '%s': %s",
                    page_num,
                    filename,
                    exc,
                )
    except Exception as exc:
        logger.error("pypdf failed to open '%s': %s", filename, exc)
        raise ValueError(f"Cannot open '{filename}': {exc}") from exc

    return pages_data, total_pages


def extract_text_from_pdf(
    file_bytes: bytes,
    filename: str,
) -> tuple[list[tuple[str, int]], int]:
    """Extract text from every page of a PDF.

    pdfplumber is tried first (superior layout handling). If it fails
    to open the file, or yields very little text, pypdf is used as a
    fallback and the richer result is kept. If both fail the PDF is
    treated as unreadable or image-only.

    Args:
        file_bytes: Raw PDF bytes.
        filename: File name used for logging context.

    Returns:
        A tuple of:
            - List of (page_text, page_number_1indexed) tuples for non-empty pages.
            - Total page count in the document.

    Raises:
        ValueError: If the PDF cannot be opened by either library.
    """
    results: list[tuple[list[tuple[str, int]], int]] = []
    pdfplumber_open_failed: Exception | None = None

    try:
        results.append(_extract_with_pdfplumber(file_bytes, filename))
    except ValueError as exc:
        results.append(([], 0))
        pdfplumber_open_failed = exc

    # Fallback when pdfplumber produced (almost) nothing.
    primary_result = results[0]
    primary_text = sum(len(t) for t, _ in primary_result[0])
    if primary_text < PDF_MIN_TEXT_LENGTH:
        try:
            fallback = _extract_with_pypdf(file_bytes, filename)
            fallback_text = sum(len(t) for t, _ in fallback[0])
            if fallback_text > primary_text:
                logger.info(
                    "pypdf fallback recovered '%s' with %d chars (pdfplumber: %d).",
                    filename,
                    fallback_text,
                    primary_text,
                )
                return fallback
        except ValueError:
            if pdfplumber_open_failed is not None and not primary_result[1]:
                # Neither library could even open the file.
                raise pdfplumber_open_failed from None
            logger.error("Both extractors failed to read '%s'.", filename)

    return primary_result


def build_documents(
    pages_data: list[tuple[str, int]],
    filename: str,
    sha256: str,
) -> list[Document]:
    """Convert raw page text into LangChain Document objects.

    Each Document carries rich metadata:
      - source: original filename
      - page: 1-indexed page number
      - sha256: file hash for dedup tracking

    Args:
        pages_data: List of (page_text, page_number) pairs.
        filename: Original PDF filename for metadata.
        sha256: File hash stored in metadata.

    Returns:
        List of LangChain Document objects, one per non-empty page.
    """
    documents: list[Document] = []
    for text, page_num in pages_data:
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_num,
                    "sha256": sha256,
                },
            )
        )
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split Document objects into overlapping chunks for embedding.

    Uses RecursiveCharacterTextSplitter which respects sentence and
    paragraph boundaries. Each chunk inherits the parent's metadata.

    Args:
        documents: Full-page Document objects.

    Returns:
        List of smaller Document chunks ready for ChromaDB.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d pages → %d chunks.", len(documents), len(chunks))
    return chunks


def process_pdf(
    file_bytes: bytes,
    filename: str,
) -> ProcessedPDF | ProcessingError:
    """Main pipeline: validate → extract → chunk → return result.

    This is the single public function called by app.py.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.
        filename: Original filename (used in metadata and UI).

    Returns:
        ProcessedPDF on success, ProcessingError on failure.
    """
    sha256 = compute_sha256(file_bytes)
    file_size_mb = len(file_bytes) / (1024 * 1024)
    is_large = False

    # ── Size warning check ──────────────────────────────────
    if file_size_mb > PDF_LARGE_SIZE_MB:
        logger.warning("'%s' is %.1f MB — flagged as large.", filename, file_size_mb)
        is_large = True

    # ── Extract text ────────────────────────────────────────
    try:
        pages_data, total_pages = extract_text_from_pdf(file_bytes, filename)
    except ValueError as exc:
        return ProcessingError(filename=filename, message=str(exc))

    if total_pages > PDF_LARGE_PAGE_COUNT:
        is_large = True

    # ── Empty / image-only PDF check ────────────────────────
    total_chars = sum(len(text) for text, _ in pages_data)
    if total_chars < PDF_MIN_TEXT_LENGTH:
        logger.warning("'%s' yielded only %d chars — likely image-only.", filename, total_chars)
        return ProcessingError(filename=filename, message=EMPTY_PDF_ERROR)

    # ── Build Document objects ──────────────────────────────
    documents = build_documents(pages_data, filename, sha256)

    # ── Chunk ───────────────────────────────────────────────
    chunks = chunk_documents(documents)

    logger.info(
        "Processed '%s': %d pages, %d chars, %d chunks, sha256=%s…",
        filename,
        total_pages,
        total_chars,
        len(chunks),
        sha256[:12],
    )

    return ProcessedPDF(
        filename=filename,
        sha256=sha256,
        page_count=total_pages,
        char_count=total_chars,
        chunks=chunks,
        is_large=is_large,
    )


def get_first_page_text(file_bytes: bytes, filename: str, max_chars: int = 3000) -> str:
    """Extract the first few thousand characters from a PDF for summarisation.

    Args:
        file_bytes: Raw PDF bytes.
        filename: Used in log messages only.
        max_chars: Maximum characters to return.

    Returns:
        Truncated plain-text string from the start of the document.
    """
    try:
        pages_data, _ = extract_text_from_pdf(file_bytes, filename)
    except ValueError:
        return ""

    combined = " ".join(text for text, _ in pages_data)
    return combined[:max_chars]
