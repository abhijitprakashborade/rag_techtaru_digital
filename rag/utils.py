"""
rag/utils.py
============
Shared utility functions: retry decorator, chat export, citation
formatting, and source-metadata helpers.

All functions are pure (no Streamlit imports) so they are fully
unit-testable in isolation.
"""

from __future__ import annotations

import html
import logging
from datetime import datetime
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Citation formatting
# ─────────────────────────────────────────────────────────────
def format_source_pills(context_docs: list[Document]) -> str:
    """Render retrieved source documents as HTML citation pills.

    Produces inline HTML <span> elements styled via the .source-pill
    CSS class injected in app.py. Deduplicates by (source, page) pair.

    Args:
        context_docs: Documents returned by the RAG chain's 'context' key.

    Returns:
        HTML string of citation pills, or empty string if no docs.
    """
    seen: set[tuple[str, int | str]] = set()
    pills: list[str] = []

    for doc in context_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = (source, page)

        if key in seen:
            continue
        seen.add(key)

        safe_source = html.escape(source)
        pills.append(
            f'<span class="source-pill">📄 {safe_source} &nbsp;·&nbsp; p.{page}</span>'
        )

    return "\n".join(pills) if pills else ""


def extract_sources(context_docs: list[Document]) -> list[dict[str, Any]]:
    """Extract structured source metadata from retrieved documents.

    Args:
        context_docs: Documents returned by the RAG chain.

    Returns:
        List of dicts with 'source' and 'page' keys (deduplicated).
    """
    seen: set[tuple[str, int | str]] = set()
    sources: list[dict[str, Any]] = []

    for doc in context_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        if (source, page) not in seen:
            seen.add((source, page))
            sources.append({"source": source, "page": page})

    return sources


# ─────────────────────────────────────────────────────────────
# Chat export
# ─────────────────────────────────────────────────────────────
def export_chat_to_text(
    messages: list[dict[str, str]],
    doc_names: list[str] | None = None,
) -> str:
    """Serialise the full conversation to a plain-text export string.

    Args:
        messages: List of {'role': str, 'content': str} session messages.
        doc_names: Optional list of uploaded document filenames for header.

    Returns:
        Formatted multi-line string suitable for .txt download.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "=" * 70,
        "  RAG-GPT — Chat with your PDF · Techtaru Digital",
        f"  Exported: {now}",
    ]

    if doc_names:
        lines.append(f"  Documents: {', '.join(doc_names)}")

    lines += ["=" * 70, ""]

    for msg in messages:
        role = "You" if msg["role"] == "user" else "Assistant"
        separator = "─" * 40
        lines.append(separator)
        lines.append(f"{role}:")
        lines.append(msg["content"])
        lines.append("")

    lines += ["=" * 70, "  End of conversation", "=" * 70]
    return "\n".join(lines)


def get_export_filename() -> str:
    """Generate a timestamped filename for the chat export.

    Returns:
        String like 'rag_chat_2024-01-15_14-30-00.txt'.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"rag_chat_{ts}.txt"


# ─────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────
def human_readable_size(size_bytes: int) -> str:
    """Convert byte count to a human-readable string.

    Args:
        size_bytes: File size in bytes.

    Returns:
        String like '2.3 MB' or '450 KB'.
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 ** 2):.1f} MB"


def truncate_text(text: str, max_chars: int = 120) -> str:
    """Truncate a string to max_chars, appending ellipsis if needed.

    Args:
        text: Input string.
        max_chars: Maximum character count including ellipsis.

    Returns:
        Truncated string.
    """
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent format for the app.

    Args:
        level: Python logging level (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
