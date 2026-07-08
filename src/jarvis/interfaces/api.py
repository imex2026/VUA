"""HTTP interface (FastAPI) sharing the same brain as CLI and voice.

The API serves one conversation - the same model as the CLI and voice
pipeline. Requests are serialized with a lock because a conversation
has strict turn order; run several instances for multi-user setups.
FastAPI/uvicorn live in the ``api`` extra.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncIterator

import structlog
from pydantic import BaseModel, Field

from jarvis.brain.orchestrator import Brain, BrainError

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = ["ChatRequest", "ChatResponse", "create_api"]


class ChatRequest(BaseModel):
    """One user message for the shared conversation."""

    message: str = Field(min_length=1, description="The user's message.")


class ChatResponse(BaseModel):
    """The assistant's complete reply."""

    reply: str


def create_api(brain: Brain, *, version: str) -> "FastAPI":
    """Build the FastAPI application around an assembled brain.

    Endpoints:
      GET  /health       liveness + version
      POST /chat         JSON in, complete reply out
      POST /chat/stream  JSON in, plain-text streamed reply out
      POST /reset        forget the conversation
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    log = structlog.get_logger("jarvis.api")
    app = FastAPI(title="Jarvis", version=version)
    turn_lock = asyncio.Lock()

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok", "version": version}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        """Answer one message and return the full reply."""
        async with turn_lock:
            parts: list[str] = []
            try:
                async for chunk in brain.respond(request.message):
                    parts.append(chunk)
            except BrainError as exc:
                log.error("api_chat_failed", error=str(exc))
                raise HTTPException(status_code=502, detail=str(exc))
        return ChatResponse(reply="".join(parts))

    @app.post("/chat/stream")
    async def chat_stream(request: ChatRequest) -> StreamingResponse:
        """Answer one message, streaming text chunks as they arrive."""

        async def generate() -> AsyncIterator[str]:
            async with turn_lock:
                try:
                    async for chunk in brain.respond(request.message):
                        yield chunk
                except BrainError as exc:
                    log.error("api_stream_failed", error=str(exc))
                    yield f"\n[error] {exc}"

        return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")

    @app.post("/reset")
    async def reset() -> dict[str, str]:
        """Clear the shared conversation history."""
        brain.reset()
        return {"status": "reset"}

    return app
