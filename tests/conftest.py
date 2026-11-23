import pytest
from langchain_core.messages import AIMessage

from papierkram.retrieval.models import Chunk


class FakeEmbeddings:
    """Returns a fixed vector. The fake store below ignores it anyway."""

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


class FakeVectorStore:
    """Holds chunks in a list; `search` returns a ranking you choose in the test."""

    def __init__(self, chunks: list[Chunk], dense_ranking: list[str] | None = None):
        self._chunks = chunks
        self._dense_ranking = dense_ranking or []

    def all_chunks(self) -> list[Chunk]:
        return self._chunks

    def search(self, query_vector: list[float], n: int) -> list[str]:
        return self._dense_ranking[:n]


class FakeLLM:
    """Records the prompt it was given and returns a canned answer."""

    def __init__(self, answer: str = "This is a test answer."):
        self.answer = answer
        self.last_prompt: str | None = None

    def invoke(self, prompt: str) -> AIMessage:
        self.last_prompt = prompt
        return AIMessage(content=self.answer)


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            id="rb-0",
            source="rundfunkbeitrag",
            chunk_index=0,
            text="Der Rundfunkbeitrag beträgt 18,36 Euro pro Monat und Wohnung.",
        ),
        Chunk(
            id="rb-1",
            source="rundfunkbeitrag",
            chunk_index=1,
            text="Studierende mit BAföG können eine Befreiung beantragen.",
        ),
        Chunk(
            id="rb-2", source="rundfunkbeitrag", chunk_index=2, text="Der Beitrag wird quartalsweise gezahlt."
        ),
        Chunk(
            id="tk-0",
            source="tk",
            chunk_index=0,
            text="Studierende sind bis 25 über die Familienversicherung versichert.",
        ),
        Chunk(
            id="anm-0",
            source="anmeldung",
            chunk_index=0,
            text="Die Anmeldung muss innerhalb von zwei Wochen nach dem Einzug erfolgen.",
        ),
    ]
