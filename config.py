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
    llm_provider: str = "mock"  # mock | openai | ollama | gemini | openrouter
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # Free LLM providers (OpenAI-compatible endpoints)
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-3.5-flash-lite"
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.2"
    temperature: float = 0.2
    max_tokens: int = 512

    # Safety
    max_upload_mb: int = 10

    # Web search (live data)
    enable_web_search: bool = True
    web_search_results: int = 3


settings = Settings()
