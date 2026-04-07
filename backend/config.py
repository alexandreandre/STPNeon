"""
Chargement et validation de la configuration applicative via pydantic-settings.
Lit les variables d'environnement depuis .env (répertoire courant ou parent) et expose
un objet Settings importable dans toute l'application.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- API (remplacement des Edge Functions Supabase) -------------------------
    # Clé OpenAI héritée de l’ancienne archi, non utilisée dans la nouvelle config.
    # On la rend optionnelle pour ne pas bloquer le chargement des settings.
    openai_api_key: str = ""
    supabase_url: str
    supabase_service_role_key: str
    supabase_anon_key: str

    # --- Ollama (ingestion / RAG hors Edge) -----------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "mistral"
    ollama_embed_model: str = "nomic-embed-text"

    # --- Qdrant ----------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "stpneon_knowledge"
    qdrant_api_key: str = ""

    # --- LLM (timeouts HTTP client : ollama, openrouter, openwebui) ------------
    # Secondes ; augmenter si Open WebUI / RAG ou modèles lents dépassent le délai.
    llm_timeout: float = 120.0

    # --- OpenRouter (LLM + embeddings via API) ---------------------------------
    openrouter_api_key: str = ""
    openrouter_site_url: str = ""
    openrouter_app_title: str = ""
    # Modèle utilisé pour le chat (LLM principal)
    # Par défaut : GPT-4o-mini via OpenRouter
    openrouter_llm_model: str = "openai/gpt-4o-mini"
    # Modèle utilisé pour les embeddings (RAG) — 1536 dims
    openrouter_embeddings_model: str = "openai/text-embedding-3-small"

    # --- Open WebUI (API compatible OpenAI, instance externe) ------------------
    # Ex. https://votre-serveur.example.com — sans slash final obligatoire
    openwebui_base_url: str = ""
    openwebui_api_key: str = ""
    # Nom du modèle tel qu’exposé par Open WebUI (sélecteur de modèles dans l’UI)
    openwebui_model: str = ""
    # Chemin relatif sous la base (par défaut l’endpoint documenté Open WebUI)
    openwebui_chat_path: str = "/api/chat/completions"
    # RAG côté Open WebUI : ID d’une collection « Knowledge » (UI) → envoyé en `files` sur /api/chat/completions
    openwebui_knowledge_collection_id: str = ""
    # Alternative avancée : JSON du tableau `files`, ex. [{"type":"collection","id":"..."},{"type":"file","id":"..."}]
    # Si non vide, il remplace openwebui_knowledge_collection_id.
    openwebui_chat_files_json: str = ""

    # --- Azure AD --------------------------------------------------------------
    azure_tenant_id: str = ""
    azure_client_id: str = ""
    azure_client_secret: str = ""

    # --- SharePoint ------------------------------------------------------------
    sharepoint_site_id: str = ""
    sharepoint_drive_id: str = ""

    # --- CORS ------------------------------------------------------------------
    allowed_origins: str = "http://localhost:5173,http://localhost:8080"

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
