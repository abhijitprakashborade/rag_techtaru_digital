"""
rag/chain.py
============
LangChain LCEL RAG chain — provider-agnostic LLM + history-aware retrieval.

Architecture (LangChain v0.3+ — NOT legacy ConversationalRetrievalChain):
  1. create_history_aware_retriever  → reformulates ambiguous follow-ups
  2. create_stuff_documents_chain    → grounded QA over retrieved chunks
  3. create_retrieval_chain          → orchestrates steps 1 + 2

Streaming yields answer tokens while capturing the retrieved context
documents, so citations come from the exact retrieval that produced
the answer (no second database round-trip).
"""

from __future__ import annotations

import logging
from typing import Any, Generator

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from tenacity import retry, stop_after_attempt, wait_exponential

from rag.config import (
    CONTEXTUALIZE_Q_SYSTEM_PROMPT,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    MEMORY_WINDOW,
    QA_SYSTEM_PROMPT,
    RETRY_ATTEMPTS,
    RETRY_WAIT_MAX,
    RETRY_WAIT_MIN,
    SUMMARY_PROMPT,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────────
def _build_llm(streaming: bool = True) -> BaseChatModel:
    """Instantiate the configured LLM provider.

    provider "gemini"    → ChatGoogleGenerativeAI (LLM_MODEL, default gemini-2.5-flash)
    provider "openai"    → ChatOpenAI       (LLM_MODEL, default gpt-4o)
    provider "anthropic" → ChatAnthropic

    Args:
        streaming: Whether to enable token-level streaming.

    Returns:
        Configured chat model instance.

    Raises:
        ValueError: If LLM_PROVIDER is not a supported value.
    """
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
            streaming=streaming,
        )

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            streaming=streaming,
        )

    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            streaming=streaming,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use 'gemini', 'openai' or 'anthropic'."
    )


# ─────────────────────────────────────────────────────────────
# Chain builder
# ─────────────────────────────────────────────────────────────
def build_rag_chain(retriever: VectorStoreRetriever, contextualize: bool = True) -> Runnable:
    """Build the full LCEL conversational RAG chain.

    Args:
        retriever: MMR-configured ChromaDB retriever.
        contextualize: When False (no chat history yet), the history-aware
            question-reformulation step is skipped entirely — that is a
            whole extra LLM round-trip per question that only pays off
            when prior turns exist.

    Returns:
        LCEL Runnable accepting {'input': str, 'chat_history': list}
        and emitting {'answer': str, 'context': list[Document]}.
    """
    llm = _build_llm(streaming=True)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    if not contextualize:
        # Fast path: retrieve + answer in one LLM call.
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        logger.info(
            "RAG chain built — provider=%s model=%s streaming=True (no-history fast path)",
            LLM_PROVIDER,
            LLM_MODEL,
        )
        return rag_chain

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    logger.info(
        "RAG chain built — provider=%s model=%s streaming=True",
        LLM_PROVIDER,
        LLM_MODEL,
    )
    return rag_chain


# ─────────────────────────────────────────────────────────────
# Memory helper
# ─────────────────────────────────────────────────────────────
def build_chat_history(raw_messages: list[dict[str, str]]) -> list[BaseMessage]:
    """Convert session messages to LangChain message objects.

    Keeps only the last MEMORY_WINDOW messages to bound token usage.

    Args:
        raw_messages: [{'role': 'user'|'assistant', 'content': str}] dicts.

    Returns:
        HumanMessage / AIMessage objects (max MEMORY_WINDOW items).
    """
    window = raw_messages[-MEMORY_WINDOW:] if len(raw_messages) > MEMORY_WINDOW else raw_messages
    history: list[BaseMessage] = []
    for msg in window:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


def _is_retryable(exc: BaseException) -> bool:
    """Auth/billing errors are permanent — retrying them just adds latency.

    Rate limits, timeouts and 5xx are transient and worth backing off on.
    """
    text = str(exc).lower()
    permanent = ("401", "403", "api key", "api key not", "invalid key", "incorrect api key", "permission", "quota exceeded", "insufficient")
    if any(mark in text for mark in permanent):
        return False
    return True


# ─────────────────────────────────────────────────────────────
# Streaming with retry + context capture
# ─────────────────────────────────────────────────────────────
@retry(
    wait=wait_exponential(min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    retry=_is_retryable,
    reraise=True,
)
def invoke_chain_with_retry(
    chain: Runnable,
    user_input: str,
    chat_history: list[BaseMessage],
) -> dict[str, Any]:
    """Invoke the RAG chain with exponential-backoff retry.

    Handles provider rate limits (429) and transient 5xx errors.
    Re-raises after RETRY_ATTEMPTS so the caller can show a friendly error.

    Args:
        chain: Built LCEL RAG chain.
        user_input: Current user question (already stripped).
        chat_history: LangChain message history objects.

    Returns:
        Dict with keys 'answer' (str) and 'context' (list[Document]).
    """
    logger.info("Invoking RAG chain. Input length=%d chars.", len(user_input))
    return chain.invoke({"input": user_input, "chat_history": chat_history})


def stream_chain(
    chain: Runnable,
    user_input: str,
    chat_history: list[BaseMessage],
    context_out: list | None = None,
) -> Generator[str, None, None]:
    """Stream the RAG chain answer token-by-token.

    The retrieval chain emits dict chunks; answer tokens are yielded
    and, when present, the 'context' documents are stashed into
    context_out. Citations therefore come from the exact retrieval
    that produced this answer.

    Retry strategy: if the provider fails before the first token (rate
    limit, timeout, network), the whole call is retried with backoff.
    Once tokens have started flowing, a partial answer is already on
    screen — retrying from scratch would duplicate text, so the error
    is raised for the UI to render.

    Args:
        chain: Built LCEL RAG chain.
        user_input: Current user question.
        chat_history: LangChain message history objects.
        context_out: Optional list the retrieved docs are appended into.

    Yields:
        Partial answer string tokens.

    Raises:
        Exception: If generation fails after tokens have started,
                   or all pre-stream retries are exhausted.
    """
    logger.info("Streaming RAG chain. Input length=%d chars.", len(user_input))

    last_error: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        buffered: list[str] = []
        try:
            for chunk in chain.stream({"input": user_input, "chat_history": chat_history}):
                if not isinstance(chunk, dict):
                    continue
                if context_out is not None and chunk.get("context"):
                    context_out.clear()
                    context_out.extend(chunk["context"])
                if "answer" in chunk:
                    token = chunk["answer"]
                    buffered.append(token)
                    yield token
            if buffered:
                logger.info("Streaming completed (%d tokens).", len(buffered))
                return
            # Empty stream with no error — treat as failure worth retrying.
            last_error = RuntimeError("Model returned an empty response.")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Stream attempt %d/%d failed: %s", attempt, RETRY_ATTEMPTS, exc
            )
            if not _is_retryable(exc):
                # Auth / billing errors are permanent — don't burn backoff time.
                logger.error("Permanent error detected; skipping retries.")
                raise
            if buffered:
                # Partial answer already displayed — do not rewind.
                logger.error("Failure after %d streamed tokens; passing to UI.", len(buffered))
                raise
        # Pre-first-token failure → backoff and try again.
        if attempt < RETRY_ATTEMPTS:
            wait = min(RETRY_WAIT_MAX, RETRY_WAIT_MIN * (2 ** (attempt - 1)))
            _sleep(wait)

    if isinstance(last_error, RuntimeError) and str(last_error) == "Model returned an empty response.":
        raise last_error
    raise last_error if last_error else RuntimeError("RAG streaming failed.")


def _sleep(seconds: int) -> None:
    """Pause between retries — split out for testability."""
    import time

    time.sleep(seconds)


# ─────────────────────────────────────────────────────────────
# Summarisation (non-RAG, direct LLM call)
# ─────────────────────────────────────────────────────────────
@retry(
    wait=wait_exponential(min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    retry=_is_retryable,
    reraise=True,
)
def generate_summary(text: str) -> str:
    """Generate a 3-bullet document summary using the LLM directly.

    Called once after upload to give users a quick overview.
    Non-streaming — rendered once in the sidebar.

    Args:
        text: First ~3000 chars of the document.

    Returns:
        Formatted bullet-point summary string.
    """
    llm = _build_llm(streaming=False)
    prompt = SUMMARY_PROMPT.format(text=text)
    logger.info("Generating document summary (%d chars).", len(text))
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(str(part) for part in content)
    return str(content)
