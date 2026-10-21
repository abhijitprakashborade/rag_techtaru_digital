"""
rag/vector_store.py
===================
ChromaDB CRUD operations — create, add, delete, query, and stats.

Design decisions:
  - Uses langchain-chroma for native LangChain integration.
  - HuggingFaceEmbeddings (all-MiniLM-L6-v2) — no second API key.
  - Persistent across restarts via CHROMA_PERSIST_DIR.
  - Collection initialisation is wrapped in try/except so startup
    never crashes on a corrupt or missing collection.
"""

from __future__ import annotations

import logging

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_METADATA,
    COLLECTION_NAME,
    EMBED_DEVICE,
    EMBED_MODEL,
    RETRIEVER_FETCH_K,
    RETRIEVER_K,
    RETRIEVER_LAMBDA_MULT,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Embedding model — cached once per Streamlit session
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading embedding model…")
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load and cache the sentence-transformer embedding model.

    Uses @st.cache_resource so the 90 MB model is downloaded and
    initialised exactly once per Streamlit server lifetime.

    Returns:
        Configured HuggingFaceEmbeddings instance.
    """
    logger.info("Initialising embedding model: %s on %s", EMBED_MODEL, EMBED_DEVICE)
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


# ─────────────────────────────────────────────────────────────
# Metadata scan cache — invalidated on every write to the store
# ─────────────────────────────────────────────────────────────
# A full metadata scan touched every chunk in the collection on
# every rerun (sidebar list + stats + chat guard = 3–4 scans per
# render). We scan once and cache until add/delete bumps the version.
_scan_cache: list[dict] | None = None


def _invalidate_scan_cache() -> None:
    """Clear the cached metadata scan after any add/delete."""
    global _scan_cache
    _scan_cache = None


# ─────────────────────────────────────────────────────────────
# Vector store lifecycle
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_cached_store() -> Chroma:
    """Open (or create) the persistent ChromaDB collection — cached.

    Cached at resource level: reopening SQLite-backed Chroma on every
    Streamlit rerun was the app's main render-latency cost. The
    returned connection stays valid across deletes and re-adds.
    """
    embeddings = get_embedding_model()

    def _init(metadata: dict | None) -> Chroma:
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_metadata=metadata,
        )

    try:
        # metadata applies at creation; existing collections keep their own.
        store = _init(COLLECTION_METADATA)
        space = (store._collection.metadata or {}).get("hnsw:space", "l2")
        if space != "cosine":
            logger.warning(
                "Collection '%s' uses hnsw:space=%s (expected cosine). "
                "Rankings still match cosine because embeddings are normalised. "
                "Delete the chroma_db/ folder to rebuild with cosine.",
                COLLECTION_NAME,
                space,
            )
        else:
            logger.info(
                "ChromaDB collection '%s' loaded (cosine space).", COLLECTION_NAME
            )
        return store
    except Exception as exc:  # noqa: BLE001
        logger.warning("ChromaDB load failed (%s). Creating fresh collection.", exc)
        return _init(None)


def get_vector_store() -> Chroma:
    """Return the process-wide cached Chroma store."""
    return _get_cached_store()


def add_documents(
    chunks: list[Document],
    progress_callback: "st.delta_generator.DeltaGenerator | None" = None,
) -> int:
    """Embed and store document chunks in ChromaDB.

    Processes chunks in batches of 50 to avoid memory spikes and
    provides optional progress updates via a Streamlit progress bar.

    Args:
        chunks: LangChain Document chunks to embed and store.
        progress_callback: Optional st.progress() widget for live updates.

    Returns:
        Number of chunks successfully stored.
    """
    store = get_vector_store()
    batch_size = 50
    total = len(chunks)
    stored = 0

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        store.add_documents(batch)
        stored += len(batch)

        if progress_callback is not None:
            progress_callback.progress(
                stored / total,
                text=f"Embedding… {stored}/{total} chunks",
            )
        logger.debug("Stored batch %d–%d of %d chunks.", i, i + len(batch), total)

    _invalidate_scan_cache()
    logger.info("add_documents complete: %d chunks stored.", stored)
    return stored


def document_exists(sha256: str) -> bool:
    """Check whether a document (by SHA-256 hash) is already indexed.

    Queries ChromaDB metadata to detect duplicates without re-embedding.

    Args:
        sha256: SHA-256 hex digest of the file to check.

    Returns:
        True if at least one chunk from this file already exists.
    """
    store = get_vector_store()
    try:
        results = store._collection.get(where={"sha256": sha256}, limit=1)
        exists = len(results.get("ids", [])) > 0
        logger.debug("SHA256 %s… exists=%s", sha256[:12], exists)
        return exists
    except Exception as exc:  # noqa: BLE001
        logger.warning("Dedup check failed: %s — assuming not exists.", exc)
        return False


def delete_document(filename: str) -> int:
    """Remove all chunks belonging to a specific source document.

    Args:
        filename: The 'source' metadata value used when chunks were added.

    Returns:
        Number of chunks deleted (may be 0 if document not found).
    """
    store = get_vector_store()
    try:
        results = store._collection.get(where={"source": filename})
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            store._collection.delete(ids=ids_to_delete)
            logger.info("Deleted %d chunks for '%s'.", len(ids_to_delete), filename)
        _invalidate_scan_cache()
        return len(ids_to_delete)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to delete '%s': %s", filename, exc)
        return 0


def _scan_metadatas() -> list[dict]:
    """Fetch all chunk metadatas (cached until the next add/delete)."""
    global _scan_cache
    if _scan_cache is not None:
        return _scan_cache
    store = get_vector_store()
    try:
        result = store._collection.get(include=["metadatas"])
        _scan_cache = [m for m in (result.get("metadatas") or []) if m]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Metadata scan failed: %s", exc)
    return _scan_cache or []


def get_collection_stats() -> dict[str, int]:
    """Return basic statistics about the current ChromaDB collection.

    Returns:
        Dict with keys 'total_chunks' and 'unique_sources'.
    """
    store = get_vector_store()
    try:
        total = store._collection.count()
        sources = {m.get("source", "") for m in _scan_metadatas()}
        return {"total_chunks": total, "unique_sources": len(sources - {""})}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch collection stats: %s", exc)
        return {"total_chunks": 0, "unique_sources": 0}


def list_indexed_sources() -> list[str]:
    """Return the list of unique source filenames currently in ChromaDB.

    Returns:
        Sorted list of unique source filenames.
    """
    try:
        sources = {m.get("source", "") for m in _scan_metadatas()}
        return sorted(sources - {""})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list sources: %s", exc)
        return []


def build_retriever() -> VectorStoreRetriever:
    """Build an MMR retriever from the current ChromaDB collection.

    MMR (Maximal Marginal Relevance) avoids returning duplicate chunks by
    balancing relevance against diversity.

    Returns:
        Configured VectorStoreRetriever using MMR search.
    """
    store = get_vector_store()
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
            "lambda_mult": RETRIEVER_LAMBDA_MULT,
        },
    )
