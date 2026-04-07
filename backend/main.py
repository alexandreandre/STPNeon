"""
Point d'entrée de l'application FastAPI (remplacement des Edge Functions Supabase).
Lance avec : cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import admin_user, chat, embed, feedback, game_questions, health, llm
from config import settings
from core.pipeline_instance import get_pipeline, _pipeline
from core.supabase_sync import sync_supabase_knowledge_to_qdrant

# Configuration logging basique si rien n'est défini (utile sur Cloud Run)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="STPNeon API", version="1.0.0")

_origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
if not _origins:
    _origins = ["http://localhost:5173", "http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    # Log de debug sur l'environnement (sans afficher les secrets)
    logger.info(
        "Startup STPNeon backend — PORT=%s, QDRANT_URL=%s, SUPABASE_URL_defined=%s, "
        "OPENROUTER_API_KEY_defined=%s",
        os.getenv("PORT"),
        os.getenv("QDRANT_URL"),
        bool(os.getenv("SUPABASE_URL")),
        bool(os.getenv("OPENROUTER_API_KEY")),
    )
    ow_base = bool((settings.openwebui_base_url or "").strip())
    ow_key = bool((settings.openwebui_api_key or "").strip())
    ow_model = bool((settings.openwebui_model or "").strip())
    logger.info(
        "Open WebUI (entrée « STPNeon OpenWebUI » dans l’UI) : %s — "
        "champs non vides base_url=%s api_key=%s model=%s",
        "oui" if (ow_base and ow_key and ow_model) else "non",
        ow_base,
        ow_key,
        ow_model,
    )
    try:
        logger.info("Initialisation du pipeline RAG (Qdrant + OpenRouter)...")
        _pipeline.init()
        logger.info("Initialisation du pipeline RAG terminée avec succès.")
        # Sync Supabase -> Qdrant au démarrage pour que le RAG ait immédiatement
        # accès aux documents déjà présents dans `knowledge_documents`.
        try:
            count = await sync_supabase_knowledge_to_qdrant(_pipeline)
            logger.info(
                "Startup — sync Supabase -> Qdrant effectuée (%d document(s) indexé(s)).",
                count,
            )
        except Exception as exc:  # pragma: no cover - dépend de l'infra externe
            logger.exception(
                "Échec de la sync Supabase -> Qdrant au démarrage (RAG utilisera une collection partielle/vides) : %s",
                exc,
            )
    except Exception as exc:  # pragma: no cover - dépend de l'infra externe
        logger.exception("Échec de l'initialisation du pipeline au démarrage : %s", exc)

app.include_router(health.router, tags=["health"])
app.include_router(chat.router, tags=["chat"])
app.include_router(embed.router, tags=["embed"])
app.include_router(admin_user.router, tags=["admin"])
app.include_router(llm.router, prefix="/api", tags=["llm"])
app.include_router(feedback.router, prefix="/api", tags=["feedback"])
app.include_router(game_questions.router, prefix="/api", tags=["assistant-game-questions"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
