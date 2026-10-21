"""
tests/test_chain_build.py
=========================
Integration-style test of the actual chain construction: the real
LCEL factory in rag/chain.py is wired to a fake chat model and an
in-memory vector store, so the full build + stream path runs with
zero network or API keys.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore

import rag.chain as rc
from rag.chain import build_rag_chain, stream_chain


@pytest.fixture
def fake_retriever():
    """MMR-style retriever over an in-memory store with fake embeddings."""
    store = InMemoryVectorStore(embedding=DeterministicFakeEmbedding(size=64))
    store.add_documents(
        [
            Document(page_content="The service fee is 2 percent.", metadata={"source": "t.pdf", "page": 1}),
            Document(page_content="Delivery is within 10 days.", metadata={"source": "t.pdf", "page": 2}),
        ]
    )
    return store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the real provider LLM with a streaming fake model."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                # 1st consumes the query-rewrite call, 2nd is the streamed answer
                AIMessage(content="standalone question: service fee"),
                AIMessage(content="The fee is covered. Source: t.pdf Page 1"),
            ]
        )
    )
    monkeypatch.setattr(rc, "_build_llm", lambda streaming=True: model)
    return model


def test_chain_builds_with_classic_factories(fake_retriever, fake_llm):
    chain = build_rag_chain(fake_retriever)
    assert chain is not None


def test_stream_yields_answer_and_captures_context(fake_retriever, fake_llm):
    chain = build_rag_chain(fake_retriever)
    context_out: list = []
    tokens = list(
        stream_chain(
            chain,
            "What is the service fee?",
            [HumanMessage(content="hello")],
            context_out,
        )
    )
    assert "".join(tokens) == "The fee is covered. Source: t.pdf Page 1"
    # The chain must have stashed the retrieved documents for citations.
    assert len(context_out) >= 1
    assert all(isinstance(d, Document) for d in context_out)


def test_stream_retries_before_first_token(fake_retriever, fake_llm, monkeypatch):
    """A provider error before the first token is retried, then succeeds."""
    monkeypatch.setattr(rc, "_sleep", lambda s: None)  # no real backoff
    chain = build_rag_chain(fake_retriever)

    class FlakyOnce:
        """Duck-typed Runnable.stream that 429s exactly once."""

        def __init__(self, inner):
            self._inner = inner
            self.failed = False

        def stream(self, payload):
            if not self.failed:
                self.failed = True
                raise RuntimeError("simulated rate limit")
            return self._inner.stream(payload)

    flaky = FlakyOnce(chain)
    tokens = list(stream_chain(flaky, "question", [], None))
    assert "".join(tokens)
    assert flaky.failed  # proves the retry path exercised


def test_stream_raises_after_retries_exhausted(fake_retriever, fake_llm, monkeypatch):
    monkeypatch.setattr(rc, "_sleep", lambda s: None)
    chain = build_rag_chain(fake_retriever)

    class AlwaysFail:
        def stream(self, payload):
            raise RuntimeError("simulated outage")

    with pytest.raises(RuntimeError):
        list(stream_chain(AlwaysFail(), "question", [], None))
