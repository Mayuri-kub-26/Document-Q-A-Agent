"""Application configuration loaded from environment variables / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings for the Document Q&A (RAG) service."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "Document Q&A Agent (RAG)"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Storage
    chroma_path: str = "data/chroma"
    collection_name: str = "documents"

    # Embeddings / chunking
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 4

    # LLM
    llm_provider: str = "mock"  # mock | openai | ollama
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.2"
    temperature: float = 0.2
    max_tokens: int = 512

    # Safety
    max_upload_mb: int = 10


settings = Settings()
