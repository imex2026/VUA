"""Long-term semantic memory: embeddings + a local vector store.

Facts extracted from conversations are embedded (sentence-transformers,
multilingual so Arabic/French/English/German all land in one space) and
stored in a persistent Chroma collection on disk. Both pieces sit
behind protocols so they can be swapped (e.g. a SQLite-VSS store) or
faked in tests. Heavy dependencies load lazily and live in the
``rag`` extra.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import structlog

__all__ = [
    "ChromaVectorStore",
    "EmbeddingBackend",
    "LongTermMemory",
    "MemoryBackendError",
    "ScoredFact",
    "SentenceTransformerEmbedder",
    "VectorStore",
]


class MemoryBackendError(RuntimeError):
    """The embedding model or vector store failed."""


@dataclass(frozen=True, slots=True)
class ScoredFact:
    """A stored fact returned from a similarity query."""

    id: str
    text: str
    score: float  # cosine similarity in [0, 1], higher = closer


class EmbeddingBackend(Protocol):
    """Anything that can embed text into vectors."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one normalized vector per input text."""
        ...


class VectorStore(Protocol):
    """Anything that can store and query embedded facts."""

    async def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Persist facts with their embeddings."""
        ...

    async def query(self, embedding: Sequence[float], top_k: int) -> list[ScoredFact]:
        """Return up to ``top_k`` nearest facts, best first."""
        ...

    async def count(self) -> int:
        """Number of stored facts."""
        ...


class SentenceTransformerEmbedder:
    """sentence-transformers embeddings, computed in a worker thread.

    The default model is multilingual, matching Jarvis's ar/fr/en/de
    persona. The model loads lazily on first use (downloads once).
    """

    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
    ) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._lock = asyncio.Lock()
        self._log = structlog.get_logger("jarvis.memory.embed")

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - env dependent
                raise MemoryBackendError(
                    "sentence-transformers is not installed. "
                    "Run: pip install -e '.[rag]'"
                ) from exc
            self._log.info("embedding_model_loading", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
            self._log.info("embedding_model_ready")
        return self._model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode ``texts`` into normalized vectors."""

        def _run() -> list[list[float]]:
            model = self._ensure_model()
            try:
                vectors = model.encode(list(texts), normalize_embeddings=True)
            except Exception as exc:
                raise MemoryBackendError(f"Embedding failed: {exc}") from exc
            result: list[list[float]] = vectors.tolist()
            return result

        async with self._lock:
            return await asyncio.to_thread(_run)


class ChromaVectorStore:
    """Persistent local vector index via Chroma (cosine space).

    All Chroma calls are synchronous, so they run in a worker thread.
    """

    def __init__(self, path: str, collection_name: str = "jarvis_memory") -> None:
        self._path = path
        self._collection_name = collection_name
        self._collection: Any = None
        self._lock = asyncio.Lock()
        self._log = structlog.get_logger("jarvis.memory.store")

    def _ensure_collection(self) -> Any:
        if self._collection is None:
            try:
                import chromadb
            except ImportError as exc:  # pragma: no cover - env dependent
                raise MemoryBackendError(
                    "chromadb is not installed. Run: pip install -e '.[rag]'"
                ) from exc
            client = chromadb.PersistentClient(path=self._path)
            self._collection = client.get_or_create_collection(
                self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._log.info(
                "vector_store_ready",
                path=self._path,
                facts=self._collection.count(),
            )
        return self._collection

    async def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Persist facts with their embeddings."""

        def _run() -> None:
            collection = self._ensure_collection()
            try:
                collection.add(
                    ids=list(ids),
                    documents=list(texts),
                    embeddings=[list(e) for e in embeddings],
                )
            except MemoryBackendError:
                raise
            except Exception as exc:
                raise MemoryBackendError(f"Store add failed: {exc}") from exc

        async with self._lock:
            await asyncio.to_thread(_run)

    async def query(self, embedding: Sequence[float], top_k: int) -> list[ScoredFact]:
        """Nearest facts by cosine similarity, best first."""

        def _run() -> list[ScoredFact]:
            collection = self._ensure_collection()
            total = collection.count()
            if total == 0:
                return []
            try:
                result = collection.query(
                    query_embeddings=[list(embedding)],
                    n_results=min(top_k, total),
                    include=["documents", "distances"],
                )
            except MemoryBackendError:
                raise
            except Exception as exc:
                raise MemoryBackendError(f"Store query failed: {exc}") from exc
            ids = result["ids"][0]
            documents = result["documents"][0]
            distances = result["distances"][0]
            return [
                # Chroma returns cosine *distance*; similarity = 1 - d.
                ScoredFact(id=i, text=t, score=1.0 - d)
                for i, t, d in zip(ids, documents, distances)
            ]

        async with self._lock:
            return await asyncio.to_thread(_run)

    async def count(self) -> int:
        """Number of stored facts."""

        def _run() -> int:
            return int(self._ensure_collection().count())

        async with self._lock:
            return await asyncio.to_thread(_run)


class LongTermMemory:
    """The memory facade the brain talks to: remember and recall.

    ``remember`` deduplicates near-identical facts so repeated mentions
    don't pile up; ``recall`` returns only facts scoring above
    ``min_score`` so unrelated memories never pollute the context.
    """

    def __init__(
        self,
        *,
        embedder: EmbeddingBackend,
        store: VectorStore,
        top_k: int = 4,
        min_score: float = 0.35,
        dedupe_score: float = 0.92,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._min_score = min_score
        self._dedupe_score = dedupe_score
        self._log = structlog.get_logger("jarvis.memory")

    async def remember(self, facts: Sequence[str]) -> int:
        """Store new facts; returns how many were actually added."""
        cleaned = [f.strip() for f in facts if f.strip()]
        if not cleaned:
            return 0
        embeddings = await self._embedder.embed(cleaned)
        added = 0
        for text, embedding in zip(cleaned, embeddings):
            existing = await self._store.query(embedding, top_k=1)
            if existing and existing[0].score >= self._dedupe_score:
                self._log.debug("fact_duplicate", fact=text, existing=existing[0].text)
                continue
            await self._store.add([uuid.uuid4().hex], [text], [embedding])
            added += 1
            self._log.info("fact_remembered", fact=text)
        return added

    async def recall(self, query: str) -> list[str]:
        """Facts relevant to ``query``, best first."""
        if not query.strip():
            return []
        embedding = (await self._embedder.embed([query]))[0]
        results = await self._store.query(embedding, top_k=self._top_k)
        relevant = [r.text for r in results if r.score >= self._min_score]
        if relevant:
            self._log.info("facts_recalled", count=len(relevant))
        return relevant
