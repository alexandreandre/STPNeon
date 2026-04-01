"""
Pipeline RAG (Retrieval-Augmented Generation) principal.
Orchestre la chaîne complète : recherche les chunks pertinents dans Qdrant →
construit le prompt avec contexte → appelle le LLM et retourne la réponse en streaming.
"""

import asyncio
import logging
from collections import defaultdict
from typing import AsyncGenerator, TypedDict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import BaseLLMProvider, LLMProviderError, get_llm_provider
from core.vector_store import QdrantStore

logger = logging.getLogger(__name__)

_MAX_HISTORY_TURNS = 10  # Nombre maximum de tours (question+réponse) conservés par conversation


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class SourceRef(TypedDict):
    filename: str
    page: str | int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_context(docs: list[Document]) -> str:
    """Formate la liste de Documents en un bloc de contexte lisible par le LLM."""
    parts: list[str] = []
    for doc in docs:
        filename = doc.metadata.get("filename") or doc.metadata.get("source", "inconnu")
        page = doc.metadata.get("page", "")
        header = f"[{filename}" + (f", page {page}]" if page else "]")
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def _extract_sources(docs: list[Document]) -> list[SourceRef]:  # noqa: F841 — conservé pour usage futur
    """Extrait les références de source uniques depuis les métadonnées des Documents."""
    seen: set[tuple] = set()
    sources: list[SourceRef] = []
    for doc in docs:
        filename = doc.metadata.get("filename") or doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        key = (filename, page)
        if key not in seen:
            seen.add(key)
            sources.append({"filename": filename, "page": page})
    return sources


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Orchestre les composants du pipeline RAG :
      - RecursiveCharacterTextSplitter (chunk_size=512, overlap=64)
      - QdrantStore pour la recherche sémantique
      - LLM provider (get_llm_provider) pour le streaming token par token

    Historique de conversation :
      Stocké en mémoire dans un dict {conversation_id: [(human, ai), ...]}.
      Limité à _MAX_HISTORY_TURNS tours (les plus anciens sont supprimés).

    Méthode publique :
      - stream_query() : tokens en streaming (SSE)
    """

    def __init__(self) -> None:
        self._store = QdrantStore()
        self._llm = get_llm_provider()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            length_function=len,
        )
        # Historique par conversation_id : list de (question, réponse)
        self._histories: dict[str, list[tuple[str, str]]] = defaultdict(list)
        logger.info("RAGPipeline initialisé.")

    # ------------------------------------------------------------------
    # Gestion de la collection Qdrant (à appeler au démarrage)
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Initialise la collection Qdrant. À appeler une fois au démarrage de l'app."""
        self._store.init_collection()

    # ------------------------------------------------------------------
    # Historique de conversation
    # ------------------------------------------------------------------

    def _get_history(self, conversation_id: str) -> list[tuple[str, str]]:
        return self._histories[conversation_id]

    def _save_turn(self, conversation_id: str, question: str, answer: str) -> None:
        """Ajoute un tour à l'historique et tronque si nécessaire."""
        history = self._histories[conversation_id]
        history.append((question, answer))
        if len(history) > _MAX_HISTORY_TURNS:
            self._histories[conversation_id] = history[-_MAX_HISTORY_TURNS:]

    def clear_history(self, conversation_id: str) -> None:
        """Efface l'historique d'une conversation (utile pour les tests ou reset UI)."""
        self._histories.pop(conversation_id, None)

    # ------------------------------------------------------------------
    # Ingestion (chunking + upsert)
    # ------------------------------------------------------------------

    async def ingest_document(self, text: str, metadata: dict) -> int:
        """
        Découpe un texte en chunks et les indexe dans Qdrant.

        Args:
            text:     Texte brut du document.
            metadata: Dict avec au minimum 'source' (et idéalement 'filename', 'page').

        Returns:
            Nombre de chunks créés et indexés.
        """
        chunks = self._splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )
        if not chunks:
            logger.warning("Aucun chunk produit pour source='%s'.", metadata.get("source"))
            return 0
        await self._store.add_documents(chunks)
        logger.info(
            "Document '%s' indexé en %d chunk(s).",
            metadata.get("source", "?"),
            len(chunks),
        )
        return len(chunks)

    # ------------------------------------------------------------------
    # Streaming token par token
    # ------------------------------------------------------------------

    async def stream_query(
        self,
        message: str,
        conversation_id: str,
        user_id: str,  # noqa: ARG002
        role_name: str = "",
        department: str = "",
        llm: BaseLLMProvider | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Interroge le pipeline RAG et yield les tokens au fur et à mesure.

          1. Récupère l'historique de la conversation
          2. Recherche sémantique dans Qdrant (k=5)
          3. Construit les messages au format OpenAI-like avec contexte + historique
          4. Stream via le LLM provider

        L'historique est mis à jour après la génération complète.
        """
        history = self._get_history(conversation_id)

        docs = await self._store.similarity_search(message, k=5)
        logger.info(
            "RAGPipeline.stream_query — conv=%s | requête='%s...' | %d document(s) de contexte.",
            conversation_id,
            (message[:80] + "…") if len(message) > 80 else message,
            len(docs),
        )
        if not docs:
            logger.warning(
                "RAGPipeline.stream_query — aucun document retourné par Qdrant pour conv=%s. "
                "Vérifier l'ingestion (chunks indexés) et la collection '%s'.",
                conversation_id,
                self._store._collection if hasattr(self._store, "_collection") else "inconnue",
            )
        else:
            preview = [
                {
                    "source": d.metadata.get("source"),
                    "filename": d.metadata.get("filename") or d.metadata.get("source"),
                    "page": d.metadata.get("page"),
                }
                for d in docs[:5]
            ]
            logger.debug("RAGPipeline.stream_query — premiers documents de contexte: %s", preview)
        # Usage d'embedding pour la requête (API /embeddings OpenRouter), si disponible
        embed_usage = self._store.get_last_embeddings_usage() or {}
        context = _format_context(docs)

        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant interne pour une société de télécom et tu réponds TOUJOURS en français. "
                    "Réponds UNIQUEMENT à partir des documents fournis ci‑dessous. "
                    "Si une information n'est pas explicitement présente dans les documents, dis-le clairement et explique ce qu'il manque.\n\n"
                    "Mise en forme attendue (Markdown) :\n"
                    "- Utilise des titres de niveau 2 (`##`) pour structurer ta réponse (par exemple `## Résumé`, `## Détails`, `## Actions recommandées`).\n"
                    "- Utilise des listes à puces pour énumérer des points.\n"
                    "- Mets en **gras** les éléments importants (décisions, chiffres clés, avertissements).\n"
                    "- Sois clair, concis et évite les phrases trop longues.\n\n"
                    "Gestion des sources :\n"
                    "- Quand tu t'appuies sur un document, mentionne-le dans le texte sous la forme `[nom_du_fichier – page X]` lorsque la page est connue.\n"
                    "- À la toute fin de ta réponse, ajoute OBLIGATOIREMENT une section `## Sources`.\n"
                    "- Dans `## Sources`, liste uniquement les documents réellement utilisés pour répondre, sous forme de liens Markdown cliquables :\n"
                    "  - `- [nom_du_fichier – page X](#)` si tu ne connais pas l'URL exacte,\n"
                    "  - ou `- [nom_du_fichier – page X](URL_COMPLÈTE)` si l'URL est présente dans le texte du contexte.\n\n"
                    f"L'utilisateur est {role_name or 'un collaborateur'} dans le département {department or 'non précisé'}.\n\n"
                    "=== DOCUMENTS PERTINENTS ===\n"
                    f"{context}"
                ),
            },
            # Historique : derniers 10 tours = 20 messages, format user/assistant alterné
            *[
                msg
                for human, ai in history[-10:]
                for msg in (
                    {"role": "user", "content": human},
                    {"role": "assistant", "content": ai},
                )
            ],
            {"role": "user", "content": message},
        ]

        active_llm = llm or self._llm
        full_response = ""
        # usage brut renvoyé par le provider (OpenRouter dans notre cas)
        raw_usage: dict[str, Any] | None = None
        try:
            async for content, usage in active_llm.stream(messages):  # type: ignore[misc]
                if usage:
                    # Dernier chunk de stream : uniquement l'usage
                    raw_usage = dict(usage)
                if content:
                    full_response += content
                    yield {"type": "token", "content": content}
        except LLMProviderError as e:
            yield {"type": "error", "content": f"\n\n⚠️ Erreur LLM : {e}"}
            return

        self._save_turn(conversation_id, message, full_response)
        logger.info(
            "stream_query() — conv=%s | %d chars | %d source(s).",
            conversation_id,
            len(full_response),
            len(docs),
        )

        # Chunk final avec les métadonnées d'usage (si disponibles)
        if raw_usage is not None:
            # Normalisation pour l'UI et le comparateur :
            # - usage.llm : tokens prompt / complétion / total
            # - usage.embeddings : tokens utilisés pour la requête (OpenRouter embeddings)
            # - usage.cost : découpe coût LLM / embeddings / total
            llm_tokens = {
                "prompt_tokens": raw_usage.get("prompt_tokens"),
                "completion_tokens": raw_usage.get("completion_tokens"),
                "total_tokens": raw_usage.get("total_tokens"),
            }
            embed_tokens = {
                "prompt_tokens": embed_usage.get("prompt_tokens"),
                "total_tokens": embed_usage.get("total_tokens"),
            }
            cost_total = raw_usage.get("cost")
            usage_struct = {
                "llm": llm_tokens,
                "embeddings": embed_tokens,
                "cost": {
                    "llm_usd": cost_total,
                    # Le endpoint /embeddings ne renvoie pas encore de coût direct ;
                    # on laisse ce champ pour évolution future.
                    "embeddings_usd": None,
                    "total_usd": cost_total,
                },
                "raw": raw_usage,
            }
            yield {"type": "meta", "usage": usage_struct}


# ---------------------------------------------------------------------------
# Tests basiques — exécuter avec : python core/rag_pipeline.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from config import settings

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    pipeline = RAGPipeline()

    async def setup() -> None:
        """Initialise Qdrant et indexe des documents de test."""
        pipeline.init()
        await pipeline.ingest_document(
            text=(
                "Le projet Telko est une plateforme de gestion documentaire interne. "
                "Elle permet d'indexer des fichiers SharePoint et de les interroger via un assistant IA."
            ),
            metadata={"source": "test-telko", "filename": "presentation_telko.pdf", "page": 1},
        )
        await pipeline.ingest_document(
            text=(
                "La politique de congés de l'entreprise prévoit 25 jours ouvrés par an. "
                "Les demandes doivent être soumises 2 semaines à l'avance via le portail RH."
            ),
            metadata={"source": "test-rh", "filename": "politique_rh.pdf", "page": 3},
        )
        await asyncio.sleep(0.5)  # Laisse Qdrant indexer
        print("Documents de test indexés.")

    async def test_stream_query() -> None:
        print("\n=== TEST stream_query() ===")
        tokens: list[str] = []
        async for token in pipeline.stream_query(
            message="Qu'est-ce que Telko ?",
            conversation_id="conv-test-001",
            user_id="user-001",
        ):
            print(token, end="", flush=True)
            tokens.append(token)
        print()
        assert len(tokens) > 0, "Le stream doit produire des tokens"
        print("OK")

    async def test_history_truncation() -> None:
        print(f"\n=== TEST troncature historique (max {_MAX_HISTORY_TURNS} tours) ===")
        conv_id = "conv-test-truncate"
        for i in range(_MAX_HISTORY_TURNS + 3):
            pipeline._save_turn(conv_id, f"question {i}", f"réponse {i}")
        history = pipeline._get_history(conv_id)
        assert len(history) == _MAX_HISTORY_TURNS, (
            f"Attendu {_MAX_HISTORY_TURNS} tours, obtenu {len(history)}"
        )
        assert history[0][0] == f"question {3}", "Les plus anciens doivent être supprimés"
        print(f"Historique tronqué à {len(history)} tours. OK")

    async def cleanup() -> None:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue
        for source_id in ("test-telko", "test-rh"):
            pipeline._store._client.delete(
                collection_name=settings.qdrant_collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(key="metadata.source", match=MatchValue(value=source_id))]
                    )
                ),
            )
        print("\nNettoyage des données de test OK.")

    async def run_all() -> None:
        await setup()
        await test_stream_query()
        await test_history_truncation()
        await cleanup()
        print("\nTous les tests sont passés.")

    asyncio.run(run_all())
