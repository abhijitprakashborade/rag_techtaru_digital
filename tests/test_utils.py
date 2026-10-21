"""
tests/test_utils.py
===================
Unit tests for rag/utils.py — citation rendering, chat export and
formatting helpers. All pure functions; no LLM or network involved.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.utils import (
    export_chat_to_text,
    format_source_pills,
    human_readable_size,
    truncate_text,
)


# ── format_source_pills ─────────────────────────────────────
class TestFormatSourcePills:
    def test_empty_context_returns_empty_string(self):
        assert format_source_pills([]) == ""

    def test_pills_render_source_and_page(self):
        docs = [Document(page_content="hello", metadata={"source": "report.pdf", "page": 3})]
        html = format_source_pills(docs)
        assert "report.pdf" in html
        assert "p.3" in html
        assert html.startswith('<span class="source-pill">')

    def test_duplicate_source_page_pairs_are_deduplicated(self):
        docs = [
            Document(page_content="a", metadata={"source": "report.pdf", "page": 1}),
            Document(page_content="b", metadata={"source": "report.pdf", "page": 1}),
            Document(page_content="c", metadata={"source": "report.pdf", "page": 2}),
        ]
        html = format_source_pills(docs)
        assert html.count("p.1") == 1
        assert html.count("p.2") == 1

    def test_hostile_source_names_are_html_escaped(self):
        docs = [Document(page_content="x", metadata={"source": "<script>alert(1)</script>", "page": 1})]
        html = format_source_pills(docs)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_missing_metadata_uses_placeholders(self):
        docs = [Document(page_content="x", metadata={})]
        html = format_source_pills(docs)
        assert "unknown" in html
        assert "p.?" in html


# ── export_chat_to_text ─────────────────────────────────────
class TestExportChat:
    def test_export_contains_roles_and_content(self):
        messages = [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "Retrieval-Augmented Generation."},
        ]
        text = export_chat_to_text(messages)
        assert "You:" in text
        assert "Assistant:" in text
        assert "What is RAG?" in text
        assert "Retrieval-Augmented Generation." in text

    def test_export_header_lists_documents(self):
        messages = [{"role": "user", "content": "hi"}]
        text = export_chat_to_text(messages, doc_names=["a.pdf", "b.pdf"])
        assert "a.pdf, b.pdf" in text

    def test_export_has_timestamp_line(self):
        text = export_chat_to_text([{"role": "user", "content": "hi"}])
        assert "Exported:" in text


# ── human_readable_size / truncate_text ─────────────────────
class TestFormatting:
    @pytest.mark.parametrize(
        "size,expected",
        [
            (500, "500 B"),
            (2048, "2.0 KB"),
            (5 * 1024 * 1024, "5.0 MB"),
        ],
    )
    def test_human_readable_size(self, size, expected):
        assert human_readable_size(size) == expected

    def test_truncate_long_text(self):
        out = truncate_text("x" * 300, max_chars=120)
        assert len(out) <= 121
        assert out.endswith("…")

    def test_truncate_keeps_short_text(self):
        assert truncate_text("short", max_chars=120) == "short"
