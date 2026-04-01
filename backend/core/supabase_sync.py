"""
Synchronisation des documents Supabase -> Qdrant.

Objectif :
  - Relire la table `knowledge_documents` de Supabase
  - Réindexer chaque document dans Qdrant via RAGPipeline.ingest_document()
  - Rendre l'opération idempotente en supprimant d'abord les anciens points pour chaque doc
"""

import logging
from typing import Any

import httpx

from config import settings
from core.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


async def sync_supabase_knowledge_to_qdrant(pipeline: RAGPipeline) -> int:
    """
    Relit tous les documents de `knowledge_documents` dans Supabase
    et les (ré)indexe dans Qdrant.

    Returns:
        Nombre de documents Supabase effectivement indexés.
    """
    if not settings.supabase_url or not (
        settings.supabase_service_role_key or settings.supabase_anon_key
    ):
        logger.warning(
            "sync_supabase_knowledge_to_qdrant — SUPABASE_URL ou clé API manquante. "
            "Sync ignorée."
        )
        return 0

    api_key = settings.supabase_service_role_key or settings.supabase_anon_key
    base_url = settings.supabase_url.rstrip("/")
    url = f"{base_url}/rest/v1/knowledge_documents"

    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    # On récupère les champs utiles uniquement
    params: dict[str, Any] = {
        "select": "id,title,content,file_path,source_type",
        "order": "id.asc",
        "limit": 5000,  # suffisant pour la plupart des cas ; à ajuster au besoin
    }

    logger.info("Sync Supabase -> Qdrant démarrée (lecture de knowledge_documents).")

    indexed_count = 0
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(url, headers=headers, params=params)
            if not resp.is_success:
                logger.error(
                    "sync_supabase_knowledge_to_qdrant — échec requête Supabase (%s) : %s",
                    resp.status_code,
                    resp.text[:500],
                )
                return 0

            rows = resp.json()
            if not isinstance(rows, list):
                logger.error(
                    "sync_supabase_knowledge_to_qdrant — réponse inattendue Supabase: %r",
                    rows,
                )
                return 0

            logger.info("Supabase a renvoyé %d document(s) à synchroniser.", len(rows))

            for row in rows:
                doc_id = row.get("id")
                content = (row.get("content") or "").strip()
                title = row.get("title") or ""
                file_path = row.get("file_path")
                source_type = row.get("source_type") or "manual"

                if not doc_id or not content:
                    # Rien à indexer pour ce document
                    continue

                source_id = f"supabase:{doc_id}"

                # On supprime d'abord les anciens points pour ce document
                try:
                    await pipeline._store.delete_document(source_id)  # type: ignore[attr-defined]
                except Exception as exc:  # pragma: no cover - dépend de Qdrant externe
                    logger.warning(
                        "sync_supabase_knowledge_to_qdrant — suppression partielle pour '%s' : %s",
                        source_id,
                        exc,
                    )

                metadata = {
                    "source": source_id,
                    "filename": title or file_path or f"doc_{doc_id}",
                    "page": 1,
                    "source_type": source_type,
                    "file_path": file_path,
                }

                try:
                    await pipeline.ingest_document(text=content, metadata=metadata)
                    indexed_count += 1
                except Exception as exc:  # pragma: no cover - dépend de Qdrant / OpenRouter
                    logger.error(
                        "sync_supabase_knowledge_to_qdrant — échec indexation doc_id=%s : %s",
                        doc_id,
                        exc,
                    )

    except Exception as exc:  # pragma: no cover - dépend IO externe
        logger.exception(
            "sync_supabase_knowledge_to_qdrant — erreur inattendue pendant la sync : %s",
            exc,
        )
        return indexed_count

    logger.info(
        "Sync Supabase -> Qdrant terminée — %d document(s) indexé(s).", indexed_count
    )
    return indexed_count

