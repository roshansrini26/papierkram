""""""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from papierkram.config import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            base_url=settings.ollama_base_url,
        )
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.llm_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
