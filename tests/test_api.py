"""Tests for the HTTP interface, driven in-process via ASGI transport."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

import httpx  # noqa: E402  (import after the fastapi gate)

from conftest import FakeLLM  # noqa: E402
from jarvis.brain.llm import LLMError  # noqa: E402
from jarvis.brain.orchestrator import Brain  # noqa: E402
from jarvis.interfaces.api import create_api  # noqa: E402
from jarvis.memory.short_term import ConversationBuffer  # noqa: E402


def _client(llm: FakeLLM) -> httpx.AsyncClient:
    brain = Brain(
        llm=llm,
        history=ConversationBuffer(max_messages=20),
        system_prompt="test",
    )
    app = create_api(brain, version="test")
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://jarvis"
    )


async def test_health() -> None:
    async with _client(FakeLLM()) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "test"}


async def test_chat_returns_full_reply() -> None:
    async with _client(FakeLLM(["Hello from HTTP."])) as client:
        response = await client.post("/chat", json={"message": "hi"})

    assert response.status_code == 200
    assert response.json() == {"reply": "Hello from HTTP."}


async def test_conversation_accumulates_across_requests() -> None:
    llm = FakeLLM(["first", "second"])
    async with _client(llm) as client:
        await client.post("/chat", json={"message": "one"})
        await client.post("/chat", json={"message": "two"})

    roles = [m.role for m in llm.calls[1]["messages"]]
    assert roles == ["user", "assistant", "user"]


async def test_reset_clears_history() -> None:
    llm = FakeLLM(["a", "b"])
    async with _client(llm) as client:
        await client.post("/chat", json={"message": "one"})
        response = await client.post("/reset")
        await client.post("/chat", json={"message": "two"})

    assert response.json() == {"status": "reset"}
    assert [m.role for m in llm.calls[1]["messages"]] == ["user"]


async def test_chat_stream_streams_text() -> None:
    async with _client(FakeLLM(["streamed reply text"])) as client:
        response = await client.post("/chat/stream", json={"message": "go"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "streamed reply text"


async def test_llm_failure_returns_502() -> None:
    llm = FakeLLM(error=LLMError("connection refused"))
    async with _client(llm) as client:
        response = await client.post("/chat", json={"message": "hi"})

    assert response.status_code == 502
    assert "language model" in response.json()["detail"]


async def test_stream_failure_reports_inline() -> None:
    llm = FakeLLM(error=LLMError("boom"))
    async with _client(llm) as client:
        response = await client.post("/chat/stream", json={"message": "hi"})

    assert response.status_code == 200  # headers already sent when it broke
    assert "[error]" in response.text


async def test_empty_message_rejected() -> None:
    async with _client(FakeLLM()) as client:
        response = await client.post("/chat", json={"message": ""})

    assert response.status_code == 422
