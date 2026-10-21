"""
tests/test_document_processor.py
================================
Unit tests for rag/document_processor.py — hashing, chunking,
document building and the price-of-entry validation rules.

Extraction itself is exercised through monkeypatched page data so
no real PDF or network is required.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

import rag.document_processor as dp
from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, EMPTY_PDF_ERROR
from rag.document_processor import (
    ProcessingError,
    build_documents,
    chunk_documents,
    compute_sha256,
    process_pdf,
)


# ── compute_sha256 ──────────────────────────────────────────
class TestSha256:
    def test_known_vector(self):
        # echo "hello" empty content, standard sha256 vector
        assert (
            compute_sha256(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_different_content_different_hash(self):
        assert compute_sha256(b"a") != compute_sha256(b"b")

    def test_hash_is_64_hex_chars(self):
        digest = compute_sha256(b"some file bytes")
        assert len(digest) == 64
        int(digest, 16)  # raises if not hexadecimal


# ── chunking ────────────────────────────────────────────────
class TestChunking:
    def test_small_page_is_not_split(self):
        docs = [Document(page_content="short text", metadata={"source": "a.pdf", "page": 1})]
        chunks = chunk_documents(docs)
        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "a.pdf"

    def test_long_page_respects_chunk_size(self):
        paragraph = "word " * 300
        text = paragraph + "\n\n" + paragraph
        docs = [Document(page_content=text, metadata={"source": "a.pdf", "page": 1})]
        chunks = chunk_documents(docs)
        assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)
        assert len(chunks) >= 2

    def test_metadata_inherited_by_chunks(self):
        page = Document(page_content="x " * 2000, metadata={"source": "b.pdf", "page": 7})
        for chunk in chunk_documents([page]):
            assert chunk.metadata["page"] == 7
            assert chunk.metadata["source"] == "b.pdf"


# ── build_documents ─────────────────────────────────────────
class TestBuildDocuments:
    def test_one_document_per_nonempty_page_with_metadata(self):
        pages = [("alpha", 1), ("beta", 2)]
        docs = build_documents(pages, "book.pdf", sha256="deadbeef")
        assert len(docs) == 2
        assert docs[0].metadata == {"source": "book.pdf", "page": 1, "sha256": "deadbeef"}
        assert docs[1].page_content == "beta"


# ── process_pdf (extraction monkeypatched) ──────────────────
def _patch_extraction(monkeypatch, pages, total=None):
    total = total if total is not None else (pages[-1][1] if pages else 0)
    monkeypatch.setattr(dp, "_extract_with_pdfplumber", lambda b, f: (pages, total))
    monkeypatch.setattr(dp, "_extract_with_pypdf", lambda b, f: ([], 0))


class TestProcessPdf:
    def test_success_returns_processed_pdf(self, monkeypatch):
        _patch_extraction(monkeypatch, [("page one text " * 50, 1)])
        result = process_pdf(b"%PDF-fake", "doc.pdf")
        assert not isinstance(result, ProcessingError)
        assert result.page_count == 1
        assert len(result.chunks) == 1
        assert result.sha256 == compute_sha256(b"%PDF-fake")

    def test_image_only_pdf_returns_processing_error(self, monkeypatch):
        _patch_extraction(monkeypatch, [("tiny", 1)])
        result = process_pdf(b"%PDF-fake", "scan.pdf")
        assert isinstance(result, ProcessingError)
        assert "image-only" in result.message or "empty" in result.message.lower()

    def test_unreadable_pdf_returns_processing_error(self, monkeypatch):
        def raise_open(b, f):
            raise ValueError("Cannot open 'x.pdf': [pypdf] nothing")

        monkeypatch.setattr(dp, "_extract_with_pdfplumber", raise_open)
        monkeypatch.setattr(dp, "_extract_with_pypdf", raise_open)
        result = process_pdf(b"not a pdf", "x.pdf")
        assert isinstance(result, ProcessingError)
        assert "Cannot open" in result.message

    def test_pypdf_fallback_recovers_low_output(self, monkeypatch):
        # pdfplumber opens the file but extracts almost nothing;
        # pypdf recovers real text and the richer result must win.
        monkeypatch.setattr(dp, "_extract_with_pdfplumber", lambda b, f: ([("", 1)], 1))
        rich = [("recovered text " * 30, 1)]
        monkeypatch.setattr(dp, "_extract_with_pypdf", lambda b, f: (rich, 1))
        result = process_pdf(b"%PDF-fake", "weird.pdf")
        assert not isinstance(result, ProcessingError)
        assert result.chunks[0].metadata["page"] == 1

    def test_large_file_flagged(self, monkeypatch):
        _patch_extraction(monkeypatch, [("x" * 200, 1)])
        # 16 MB of raw bytes → above PDF_LARGE_SIZE_MB
        big = b"A" * 16 * 1024 * 1024
        result = process_pdf(big, "big.pdf")
        assert isinstance(result, dp.ProcessedPDF)
        assert result.is_large is True

    def test_error_message_matches_config_text(self, monkeypatch):
        _patch_extraction(monkeypatch, [("", 1)])
        result = process_pdf(b"%PDF-fake", "s.pdf")
        assert result.message == EMPTY_PDF_ERROR
