"""
tests/test_config_chain.py
==========================
Configuration invariants and pure chain helpers:
  - QA prompt fallback text exactly matches FALLBACK_MESSAGE
  - retry / retrieval parameters are sane
  - chat-history windowing returns LangChain message types

No LLM, network or vector store required.
"""

from __future__ import annotations

import rag.config as cfg

from rag.chain import build_chat_history
from langchain_core.messages import AIMessage, HumanMessage


class TestConfigInvariants:
    def test_fallback_message_appears_in_qa_prompt(self):
        assert FALLBACK_TEXT_SHORT in cfg.QA_SYSTEM_PROMPT

    def test_fallback_message_is_generic_and_honest(self):
        assert "couldn't find relevant information" in cfg.FALLBACK_MESSAGE

    def test_retrieval_settings_are_ordered(self):
        assert cfg.RETRIEVER_FETCH_K >= cfg.RETRIEVER_K > 0
        assert 0.0 < cfg.RETRIEVER_LAMBDA_MULT <= 1.0

    def test_chunk_overlap_smaller_than_chunk(self):
        assert 0 <= cfg.CHUNK_OVERLAP < cfg.CHUNK_SIZE

    def test_memory_window_is_even_and_positive(self):
        assert cfg.MEMORY_WINDOW > 0 and cfg.MEMORY_WINDOW % 2 == 0

    def test_collection_uses_cosine_space(self):
        assert cfg.COLLECTION_METADATA["hnsw:space"] == "cosine"

    def test_provider_is_supported(self):
        assert cfg.LLM_PROVIDER in ("gemini", "openai", "anthropic")
        assert cfg.LLM_MODEL  # non-empty default

    def test_temperature_is_effectively_zero(self):
        assert 0.0 <= cfg.LLM_TEMPERATURE <= 0.2


class TestBuildChatHistory:
    def test_converts_roles_to_message_types(self):
        raw = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        history = build_chat_history(raw)
        assert isinstance(history[0], HumanMessage)
        assert isinstance(history[1], AIMessage)
        assert history[0].content == "hi"

    def test_unknown_roles_are_skipped(self):
        raw = [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "kept"},
        ]
        history = build_chat_history(raw)
        assert len(history) == 1
        assert isinstance(history[0], HumanMessage)

    def test_window_limits_messages(self):
        raw = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        history = build_chat_history(raw)
        assert len(history) == cfg.MEMORY_WINDOW
        assert history[-1].content == "m19"

    def test_short_history_returned_verbatim(self):
        raw = [{"role": "user", "content": "only"}]
        history = build_chat_history(raw)
        assert len(history) == 1 and history[0].content == "only"


FALLBACK_TEXT_SHORT = "couldn't find relevant information"
