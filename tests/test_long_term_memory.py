"""Unit tests for the long-term memory facade (fake embedder + store)."""

from __future__ import annotations

import math
import uuid
from typing import Sequence

from jarvis.memory.long_term import LongTermMemory, ScoredFact


class FakeEmbedder:
    """Maps texts to fixed vectors; unknown texts get an orthogonal one."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.table.get(t, [0.0, 0.0, 1.0]) for t in texts]


class InMemoryStore:
    """A tiny cosine-similarity store, mirroring the VectorStore protocol."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, list[float]]] = []

    async def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        for i, t, e in zip(ids, texts, embeddings):
            self.rows.append((i, t, list(e)))

    async def query(self, embedding: Sequence[float], top_k: int) -> list[ScoredFact]:
        def cosine(a: Sequence[float], b: Sequence[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
            return dot / norm if norm else 0.0

        scored = [
            ScoredFact(id=i, text=t, score=cosine(embedding, e))
            for i, t, e in self.rows
        ]
        scored.sort(key=lambda f: f.score, reverse=True)
        return scored[:top_k]

    async def count(self) -> int:
        return len(self.rows)


def _memory(
    table: dict[str, list[float]], store: InMemoryStore | None = None
) -> tuple[LongTermMemory, InMemoryStore]:
    store = store or InMemoryStore()
    memory = LongTermMemory(
        embedder=FakeEmbedder(table),
        store=store,
        top_k=3,
        min_score=0.5,
        dedupe_score=0.95,
    )
    return memory, store


async def test_remember_and_recall_relevant_fact() -> None:
    table = {
        "The user's name is Aymen.": [1.0, 0.0, 0.0],
        "what is my name?": [0.9, 0.1, 0.0],  # close to the fact
    }
    memory, store = _memory(table)

    added = await memory.remember(["The user's name is Aymen."])
    assert added == 1
    assert await store.count() == 1

    facts = await memory.recall("what is my name?")
    assert facts == ["The user's name is Aymen."]


async def test_recall_filters_low_scores() -> None:
    table = {
        "The user likes green tea.": [1.0, 0.0, 0.0],
        "how do rockets work?": [0.0, 1.0, 0.0],  # orthogonal
    }
    memory, _ = _memory(table)
    await memory.remember(["The user likes green tea."])

    assert await memory.recall("how do rockets work?") == []


async def test_remember_dedupes_near_identical_facts() -> None:
    table = {
        "The user lives in Tunis.": [1.0, 0.0, 0.0],
        "User lives in Tunis city.": [0.99, 0.01, 0.0],  # ~same vector
    }
    memory, store = _memory(table)

    assert await memory.remember(["The user lives in Tunis."]) == 1
    assert await memory.remember(["User lives in Tunis city."]) == 0
    assert await store.count() == 1


async def test_remember_skips_blank_facts() -> None:
    memory, store = _memory({})

    assert await memory.remember(["", "   ", "\n"]) == 0
    assert await store.count() == 0


async def test_recall_empty_query_returns_nothing() -> None:
    memory, _ = _memory({})
    assert await memory.recall("   ") == []


async def test_recall_orders_best_first_and_caps_top_k() -> None:
    store = InMemoryStore()
    # Pre-load four facts at decreasing similarity to the query axis.
    vectors = [
        ("closest", [1.0, 0.0, 0.0]),
        ("close", [0.9, 0.44, 0.0]),
        ("medium", [0.7, 0.71, 0.0]),
        ("far", [0.55, 0.84, 0.0]),
    ]
    for text, vector in vectors:
        await store.add([uuid.uuid4().hex], [text], [vector])
    memory = LongTermMemory(
        embedder=FakeEmbedder({"q": [1.0, 0.0, 0.0]}),
        store=store,
        top_k=3,
        min_score=0.5,
    )

    facts = await memory.recall("q")

    assert facts == ["closest", "close", "medium"]  # far cut by top_k
