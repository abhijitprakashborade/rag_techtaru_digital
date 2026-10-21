"""
rag/config.py
============
Central configuration for the RAG Chat PDF application.

Every constant, prompt and user-facing message lives here. Runtime the
LLM provider (and key) is resolved from environment variables so the
same codebase runs against OpenAI or Anthropic without code changes.
"""

from __future__ import annotations

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR: str = str(BASE_DIR / "chroma_db")

# ─────────────────────────────────────────────────────────────
# ChromaDB
# ─────────────────────────────────────────────────────────────
COLLECTION_NAME: str = "techtaru_pdf_rag"
# Cosine distance matches how we normalise embeddings and is the
# metric most people intuitively expect from semantic search.
COLLECTION_METADATA: dict[str, str] = {"hnsw:space": "cosine"}

# ─────────────────────────────────────────────────────────────
# Embedding Model (local — no second API key)
# ─────────────────────────────────────────────────────────────
EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DEVICE: str = "cpu"

# ─────────────────────────────────────────────────────────────
# LLM — provider resolved from environment
#   LLM_PROVIDER = "gemini" (free tier) | "openai" | "anthropic"
#   GOOGLE_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
# ─────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

_DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
}
LLM_MODEL: str = os.getenv("LLM_MODEL", "").strip() or _DEFAULT_MODELS.get(
    LLM_PROVIDER, "gemini-2.5-flash"
)
LLM_MAX_TOKENS: int = 2048
LLM_TEMPERATURE: float = 0.1

# ─────────────────────────────────────────────────────────────
# Document chunking
# ─────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 150

# ─────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────
RETRIEVER_K: int = 4
RETRIEVER_FETCH_K: int = 20
RETRIEVER_LAMBDA_MULT: float = 0.7   # MMR diversity parameter

# ─────────────────────────────────────────────────────────────
# Memory — how many prior messages to keep (6 = 3 full turns)
# ─────────────────────────────────────────────────────────────
MEMORY_WINDOW: int = 6

# ─────────────────────────────────────────────────────────────
# PDF constraints
# ─────────────────────────────────────────────────────────────
PDF_MIN_TEXT_LENGTH: int = 100        # chars — below this = image-only
PDF_LARGE_SIZE_MB: float = 15.0       # warn user above this
PDF_LARGE_PAGE_COUNT: int = 150       # warn user above this

# ─────────────────────────────────────────────────────────────
# Tenacity retry settings
# ─────────────────────────────────────────────────────────────
RETRY_WAIT_MIN: int = 2
RETRY_WAIT_MAX: int = 10
RETRY_ATTEMPTS: int = 3

# ─────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────
CONTEXTUALIZE_Q_SYSTEM_PROMPT: str = (
    "Given the chat history and the latest user question, rewrite the "
    "question as a complete standalone question that needs no prior context. "
    "Return ONLY the reformulated question — nothing else. "
    "If the question is already standalone, return it unchanged."
)

QA_SYSTEM_PROMPT: str = """You are a precise document analyst. You answer questions \
STRICTLY based on the provided document context.

Rules:
1. Answer ONLY using information from the context below — never from prior training knowledge.
2. If the answer is not found in the context, respond with exactly this line:
   "I couldn't find relevant information in the uploaded documents to answer that. Please try rephrasing or upload a more relevant document."
3. Always cite the source document name and page number at the end of your answer using this format:
   Source: <filename> · Page <page_number>
4. Be concise and factual. No filler phrases.
5. Use markdown formatting where helpful (bullet lists, bold key terms).

Context:
{context}"""

SUMMARY_PROMPT: str = """Provide a concise 3-bullet summary of the key topics covered in this text. \
Be specific — mention actual facts, names, dates or figures if they appear.

Format:
• <point 1>
• <point 2>
• <point 3>

Text:
{text}"""

# ─────────────────────────────────────────────────────────────
# Fallback / UI strings (single source of truth)
# ─────────────────────────────────────────────────────────────
FALLBACK_MESSAGE: str = (
    "I couldn't find relevant information in the uploaded documents to answer that. "
    "Please try rephrasing or upload a more relevant document."
)

NO_DOCS_WARNING: str = "📂 Please upload at least one PDF before asking questions."
EMPTY_INPUT_WARNING: str = "⌨️ Please type a question before submitting."
EMPTY_PDF_ERROR: str = (
    "❌ This PDF appears to be empty or image-only (scanned). "
    "Neither pdfplumber nor pypdf could extract usable text. "
    "Please upload a text-based PDF."
)
LARGE_FILE_WARNING: str = (
    "⚠️ Large file detected — processing may take a while. "
    "Embedding runs in batches with live progress."
)
DUPLICATE_FILE_INFO: str = "ℹ️ **{filename}** was already uploaded and indexed. Skipping re-embedding."

APP_TITLE: str = "RAG-GPT"
APP_SUBTITLE: str = "Chat with your PDF · LangChain LCEL · ChromaDB"
APP_ICON: str = "✨"


# Keep the QA prompt and the standalone fallback message in sync.
assert FALLBACK_MESSAGE.split(".")[0].lower() in QA_SYSTEM_PROMPT.lower(), (
    "QA_SYSTEM_PROMPT fallback text must match FALLBACK_MESSAGE"
)
