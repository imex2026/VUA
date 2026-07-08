"""Unit tests for the brain's tool-use loop, with the LLM mocked."""

from __future__ import annotations

from typing import Any

from conftest import FakeLLM
from jarvis.brain.llm import ResponseComplete, TextDelta, ToolUseRequest
from jarvis.brain.orchestrator import Brain
from jarvis.events import EventBus, ToolCallCompleted, ToolCallRequested
from jarvis.memory.short_term import ConversationBuffer
from jarvis.tools.base import ToolError, ToolRegistry

DONE = ResponseComplete(stop_reason="tool_use", input_tokens=1, output_tokens=1)


class EchoTool:
    name = "echo"
    description = "Echo text."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    async def run(self, arguments: dict[str, Any]) -> str:
        self.calls.append(arguments)
        if self.fail:
            raise ToolError("echo is broken")
        return f"echo: {arguments['text']}"


def _brain(
    llm: FakeLLM,
    tool: EchoTool,
    bus: EventBus | None = None,
    max_tool_iterations: int = 8,
) -> Brain:
    registry = ToolRegistry()
    registry.register(tool)
    return Brain(
        llm=llm,
        history=ConversationBuffer(max_messages=40),
        system_prompt="test",
        bus=bus,
        tools=registry,
        max_tool_iterations=max_tool_iterations,
    )


def _tool_call_reply(text: str, call_id: str, args: dict[str, Any]) -> list[Any]:
    events: list[Any] = []
    if text:
        events.append(TextDelta(text))
    events.append(ToolUseRequest(id=call_id, name="echo", arguments=args))
    events.append(DONE)
    return events


async def test_tool_round_trip() -> None:
    llm = FakeLLM(
        [
            _tool_call_reply("Let me check. ", "t1", {"text": "hi"}),
            "The echo said hi.",
        ]
    )
    tool = EchoTool()
    brain = _brain(llm, tool)

    chunks = [c async for c in brain.respond("run the echo")]

    assert "".join(chunks) == "Let me check. The echo said hi."
    assert tool.calls == [{"text": "hi"}]

    # The second LLM call saw the full tool exchange.
    messages = llm.calls[1]["messages"]
    roles = [m.role for m in messages]
    assert roles == ["user", "assistant", "user"]
    assistant_blocks = messages[1].content
    assert assistant_blocks[0] == {"type": "text", "text": "Let me check. "}
    assert assistant_blocks[1]["type"] == "tool_use"
    assert assistant_blocks[1]["id"] == "t1"
    tool_result = messages[2].content[0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "t1"
    assert tool_result["content"] == "echo: hi"
    assert tool_result["is_error"] is False

    # Tool specs were sent to the model.
    assert llm.calls[0]["tools"][0]["name"] == "echo"


async def test_tool_events_published() -> None:
    bus = EventBus()
    requested: list[ToolCallRequested] = []
    completed: list[ToolCallCompleted] = []

    async def on_requested(event: ToolCallRequested) -> None:
        requested.append(event)

    async def on_completed(event: ToolCallCompleted) -> None:
        completed.append(event)

    bus.subscribe(ToolCallRequested, on_requested)
    bus.subscribe(ToolCallCompleted, on_completed)

    llm = FakeLLM([_tool_call_reply("", "t9", {"text": "ping"}), "done"])
    brain = _brain(llm, EchoTool(), bus)

    async for _ in brain.respond("go"):
        pass
    await bus.join()

    assert len(requested) == 1 and requested[0].tool_name == "echo"
    assert len(completed) == 1 and completed[0].result == "echo: ping"
    assert completed[0].call_id == "t9"


async def test_failing_tool_reported_to_model() -> None:
    llm = FakeLLM([_tool_call_reply("", "t2", {"text": "x"}), "Sorry, echo is down."])
    brain = _brain(llm, EchoTool(fail=True))

    chunks = [c async for c in brain.respond("try it")]

    assert "".join(chunks) == "Sorry, echo is down."
    tool_result = llm.calls[1]["messages"][2].content[0]
    assert tool_result["is_error"] is True
    assert tool_result["content"] == "echo is broken"


async def test_tool_iteration_limit() -> None:
    # The model keeps asking for tools forever; the brain must stop it.
    replies = [_tool_call_reply("", f"t{i}", {"text": str(i)}) for i in range(10)]
    llm = FakeLLM(replies)
    tool = EchoTool()
    brain = _brain(llm, tool, max_tool_iterations=2)

    chunks = [c async for c in brain.respond("loop forever")]

    assert len(tool.calls) == 2
    assert len(llm.calls) == 2
    assert "tool calls" in "".join(chunks)  # the limit note was yielded


async def test_no_tools_sends_none() -> None:
    llm = FakeLLM(["plain answer"])
    brain = Brain(
        llm=llm,
        history=ConversationBuffer(max_messages=10),
        system_prompt="test",
        tools=None,
    )

    async for _ in brain.respond("hello"):
        pass

    assert llm.calls[0]["tools"] is None
