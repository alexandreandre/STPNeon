"""
Interface avec Qdrant (vector store).
Gère la création de la collection, l'upsert de points (chunks + metadata),
la recherche par similarité cosine et la suppression de points par document ID.
"""

import asyncio
import logging
from typing import Any

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    VectorParams,
)

from config import settings

logger = logging.getLogger(__name__)

_VECTOR_SIZE = 1536  # Doit correspondre à la dimension du modèle d'embedding (ex: text-embedding-3-small)


class OpenRouterEmbeddings(Embeddings):
    """
    Implémentation minimale de l'interface Embeddings de LangChain
    en appelant l'API /embeddings d'OpenRouter (full cloud).
    """

    def __init__(self) -> None:
        self._api_key = settings.openrouter_api_key
        self._model = settings.openrouter_embeddings_model
        self._base_url = "https://openrouter.ai/api/v1/embeddings"
        # Dernière métrique d'usage renvoyée par OpenRouter (prompt_tokens, total_tokens, etc.).
        # Utilisée pour alimenter les statistiques RAG (tokens embeddings).
        self._last_usage: dict[str, Any] | None = None

        if not self._api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY manquant. Renseignez-le dans les variables d'environnement."
            )

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if settings.openrouter_site_url:
            headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_title:
            headers["X-OpenRouter-Title"] = settings.openrouter_app_title
        return headers

    def _request_embeddings(self, inputs: list[str]) -> list[list[float]]:
        import httpx

        payload = {
            "model": self._model,
            "input": inputs,
        }
        try:
            resp = httpx.post(self._base_url, headers=self._build_headers(), json=payload, timeout=60.0)
        except httpx.RequestError as exc:
            # En dev ou si OpenRouter est lent/indisponible, on ne veut pas casser tout le backend.
            logger.error(
                "OpenRouterEmbeddings._request_embeddings → erreur réseau vers OpenRouter: %s. "
                "Fallback vers vecteurs nuls dim=%d pour %d texte(s).",
                exc,
                _VECTOR_SIZE,
                len(inputs),
            )
            return [[0.0] * _VECTOR_SIZE for _ in inputs]

        if resp.status_code != 200:
            logger.error(
                "OpenRouterEmbeddings._request_embeddings → HTTP %d: %s",
                resp.status_code,
                resp.text[:500],
            )
            raise RuntimeError(f"Erreur OpenRouter embeddings ({resp.status_code}) : {resp.text}")

        data = resp.json()
        # OpenRouter / OpenAI style : usage = { prompt_tokens, total_tokens, ... }
        usage = data.get("usage")
        if isinstance(usage, dict):
            self._last_usage = usage
        else:
            self._last_usage = None
        items = data.get("data", [])
        if not items:
            # Log détaillé pour comprendre pourquoi aucun embedding n'est renvoyé
            logger.warning(
                "OpenRouterEmbeddings._request_embeddings → aucune entrée dans data[]. "
                "meta=%s, error=%s, usage=%s, inputs_count=%d, model=%s",
                data.get("meta"),
                data.get("error") or data.get("message"),
                data.get("usage"),
                len(inputs),
                self._model,
            )
        # OpenAI / OpenRouter compatible : data -> { data: [ { embedding: [...] }, ... ] }
        return [item["embedding"] for item in items]

    def get_last_usage(self) -> dict[str, Any] | None:
        """
        Retourne la dernière métrique d'usage renvoyée par /embeddings.

        Typiquement :
          { "prompt_tokens": int, "total_tokens": int, ... }
        """
        return self._last_usage

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._request_embeddings(texts)
        if not vectors or not vectors[0]:
            logger.warning(
                "OpenRouterEmbeddings.embed_documents → aucun vecteur utile renvoyé "
                "(model=%s, nb_texts=%d). Fallback vers vecteurs nuls dim=%d.",
                self._model,
                len(texts),
                _VECTOR_SIZE,
            )
            # Fallback tolérant : retourne des vecteurs nuls de la bonne dimension
            return [[0.0] * _VECTOR_SIZE for _ in texts]

        dim = len(vectors[0])
        if dim != _VECTOR_SIZE:
            logger.warning(
                "OpenRouterEmbeddings.embed_documents → dimension inattendue %d (attendu=%d). "
                "Fallback vers vecteurs nuls.",
                dim,
                _VECTOR_SIZE,
            )
            return [[0.0] * _VECTOR_SIZE for _ in texts]

        logger.debug(
            "OpenRouterEmbeddings.embed_documents → %d vecteur(s), dim=%d (model=%s)",
            len(vectors),
            dim,
            self._model,
        )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        logger.debug(
            "OpenRouterEmbeddings.embed_query → texte longueur=%d, aperçu='%s'",
            len(text),
            (text[:80] + "…") if len(text) > 80 else text,
        )
        vectors = self._request_embeddings([text])
        if not vectors or not vectors[0]:
            logger.warning(
                "OpenRouterEmbeddings.embed_query → vecteur vide ou manquant "
                "(model=%s, texte_longueur=%d). Fallback vers vecteur nul dim=%d.",
                self._model,
                len(text),
                _VECTOR_SIZE,
            )
            return [0.0] * _VECTOR_SIZE

        dim = len(vectors[0])
        if dim != _VECTOR_SIZE:
            logger.warning(
                "OpenRouterEmbeddings.embed_query → dimension inattendue %d (attendu=%d). "
                "Fallback vers vecteur nul.",
                dim,
                _VECTOR_SIZE,
            )
            return [0.0] * _VECTOR_SIZE

        logger.debug(
            "OpenRouterEmbeddings.embed_query → dim=%d (model=%s)",
            dim,
            self._model,
        )
        return vectors[0]


class QdrantStore:
    """
    Wrapper autour de Qdrant et langchain_qdrant.QdrantVectorStore.

    Expose :
      - init_collection() : crée la collection si absente (cosine, 768 dims)
      - add_documents()   : embed + upsert des chunks LangChain Document
      - similarity_search(): recherche sémantique avec filtre optionnel
      - delete_document() : suppression par metadata.source
      - as_retriever()    : retourne un BaseRetriever LangChain pour les chains

    Paramètres lus depuis config.settings :
      - qdrant_url              : URL de l'instance Qdrant
      - qdrant_collection_name  : nom de la collection
      - ollama_base_url         : URL Ollama pour les embeddings
      - ollama_embed_model      : modèle d'embedding
    """

    def __init__(self) -> None:
        self._collection = settings.qdrant_collection_name
        logger.info(
            "QdrantStore.__init__ — collection=%s, url=%s, api_key_defined=%s",
            self._collection,
            settings.qdrant_url,
            bool(settings.qdrant_api_key),
        )

        # Instance Qdrant externe (URL + API key) pour dev et prod full cloud.
        try:
            self._client = QdrantClient(
                url=settings.qdrant_url,
                # Important pour les déploiements "managés" (Cloud Run, proxies, etc.) :
                # si on ne précise pas port=None, le client ajoute par défaut :6333,
                # ce qui provoque des timeouts sur une URL HTTPS publique sans ce port.
                port=None,
                api_key=settings.qdrant_api_key or None,
            )
        except Exception as exc:  # pragma: no cover - dépend de l'infra externe
            logger.exception("Échec de la création du client QdrantClient : %s", exc)
            raise

        # Embeddings full cloud via OpenRouter (API /embeddings).
        try:
            self._embeddings = OpenRouterEmbeddings()
        except Exception as exc:  # pragma: no cover - dépend de l'infra externe
            logger.exception("Échec de l'initialisation d'OpenRouterEmbeddings : %s", exc)
            raise

        # Lazy : initialisé après init_collection()
        self._lc_store: QdrantVectorStore | None = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_collection(self) -> None:
        """
        Crée la collection Qdrant si elle n'existe pas encore.
        Peut être appelée au démarrage OU à la volée (lazy init).
        Distance : cosine. Taille des vecteurs : _VECTOR_SIZE.
        """
        if self._lc_store is not None:
            # Déjà initialisé (ex: appel précédent depuis startup ou lazy init)
            return

        logger.info(
            "QdrantStore.init_collection — tentative d'initialisation de la collection '%s'.",
            self._collection,
        )
        try:
            existing = {c.name for c in self._client.get_collections().collections}
            if self._collection not in existing:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
                )
                logger.info(
                    "Collection '%s' créée (dim=%d, cosine).",
                    self._collection,
                    _VECTOR_SIZE,
                )
            else:
                logger.info("Collection '%s' déjà existante.", self._collection)

            # Initialise le store LangChain maintenant que la collection existe
            self._lc_store = QdrantVectorStore(
                client=self._client,
                collection_name=self._collection,
                embedding=self._embeddings,
            )
        except Exception as exc:  # pragma: no cover - dépend de l'infra externe
            logger.exception("Échec de l'initialisation de la collection Qdrant : %s", exc)
            # On laisse _lc_store à None pour que les appels suivants puissent retenter
            raise

    def _require_store(self) -> QdrantVectorStore:
        """Lève une erreur explicite si init_collection() n'a pas été appelé."""
        if self._lc_store is None:
            raise RuntimeError(
                "QdrantStore non initialisé. Appelez init_collection() au démarrage."
            )
        return self._lc_store

    # ------------------------------------------------------------------
    # Écriture
    # ------------------------------------------------------------------

    async def add_documents(self, docs: list[Document]) -> None:
        """
        Embed et insère une liste de Documents LangChain dans Qdrant.

        Chaque Document doit avoir :
          - page_content : le texte du chunk
          - metadata     : dict avec au minimum 'source' (et idéalement 'filename', 'page')

        Args:
            docs: Liste de langchain_core.documents.Document.

        Raises:
            RuntimeError: Si Qdrant n'est pas joignable ou si l'upsert échoue.
            ValueError:   Si la liste est vide.
        """
        if not docs:
            raise ValueError("La liste de documents ne peut pas être vide.")

        # Lazy init : si Qdrant n'a pas été prêt au startup, on retente ici
        try:
            self.init_collection()
        except Exception as exc:  # pragma: no cover - dépend de l'infra externe
            raise RuntimeError(f"Erreur Qdrant (init_collection dans add_documents) : {exc}") from exc

        store = self._require_store()
        logger.info("Upsert de %d document(s) dans '%s'.", len(docs), self._collection)
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, store.add_documents, docs
            )
        except Exception as exc:
            logger.error("Erreur lors de l'upsert dans Qdrant : %s", exc)
            raise RuntimeError(f"Erreur Qdrant (add_documents) : {exc}") from exc

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Filter | None = None,
    ) -> list[Document]:
        """
        Recherche les k documents les plus proches du vecteur de la requête.

        Args:
            query:  Texte de la question utilisateur.
            k:      Nombre de résultats à retourner.
            filter: Filtre Qdrant optionnel (qdrant_client.models.Filter).

        Returns:
            Liste de Documents LangChain ordonnés par similarité décroissante.

        Raises:
            RuntimeError: Si la recherche échoue.
        """
        # Lazy init : si Qdrant n'a pas été prêt au startup, on retente ici
        try:
            self.init_collection()
        except Exception as exc:  # pragma: no cover - dépend de l'infra externe
            raise RuntimeError(f"Erreur Qdrant (init_collection dans similarity_search) : {exc}") from exc

        store = self._require_store()
        logger.debug(
            "QdrantStore.similarity_search → début (len(query)=%d, k=%d, collection=%s)",
            len(query),
            k,
            self._collection,
        )
        search_kwargs: dict[str, Any] = {"k": k}
        if filter is not None:
            search_kwargs["filter"] = filter

        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: store.similarity_search(query, **search_kwargs),
            )
            logger.debug("similarity_search → %d résultat(s) pour '%s...'", len(results), query[:60])
            return results
        except Exception as exc:
            logger.error("Erreur lors de la recherche Qdrant : %s", exc)
            raise RuntimeError(f"Erreur Qdrant (similarity_search) : {exc}") from exc

    def get_last_embeddings_usage(self) -> dict[str, Any] | None:
        """
        Expose l'usage du dernier appel /embeddings effectué via OpenRouterEmbeddings.

        Utilisé par la pipeline RAG pour remonter les tokens d'embedding par requête.
        """
        return self._embeddings.get_last_usage()

    # ------------------------------------------------------------------
    # Suppression
    # ------------------------------------------------------------------

    async def delete_document(self, source_id: str) -> None:
        """
        Supprime tous les points dont metadata.source == source_id.

        Args:
            source_id: Valeur du champ 'source' dans les métadonnées du document
                       (ex : chemin SharePoint, UUID document, nom de fichier).

        Raises:
            RuntimeError: Si la suppression échoue.
        """
        logger.info("Suppression des points avec source='%s'.", source_id)
        # Lazy init : si Qdrant n'a pas été prêt au startup, on retente ici
        try:
            self.init_collection()
        except Exception as exc:  # pragma: no cover - dépend de l'infra externe
            raise RuntimeError(f"Erreur Qdrant (init_collection dans delete_document) : {exc}") from exc
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.delete(
                    collection_name=self._collection,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="metadata.source",
                                    match=MatchValue(value=source_id),
                                )
                            ]
                        )
                    ),
                ),
            )
        except Exception as exc:
            logger.error("Erreur lors de la suppression dans Qdrant : %s", exc)
            raise RuntimeError(f"Erreur Qdrant (delete_document) : {exc}") from exc

    # ------------------------------------------------------------------
    # Intégration LangChain
    # ------------------------------------------------------------------

    def as_retriever(self, k: int = 5):
        """
        Retourne un BaseRetriever LangChain compatible avec les chains
        (ConversationalRetrievalChain, etc.).

        Args:
            k: Nombre de documents à retourner par recherche.
        """
        return self._require_store().as_retriever(search_kwargs={"k": k})


# ---------------------------------------------------------------------------
# Tests basiques — exécuter avec : python core/vector_store.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    store = QdrantStore()

    async def test_init_collection() -> None:
        print("\n=== TEST init_collection() ===")
        store.init_collection()
        collections = store._client.get_collections().collections
        names = [c.name for c in collections]
        assert settings.qdrant_collection_name in names, "La collection doit exister"
        print(f"Collections présentes : {names}")
        print("OK")

    async def test_add_and_search() -> None:
        print("\n=== TEST add_documents() + similarity_search() ===")
        docs = [
            Document(
                page_content="Ollama permet de faire tourner des LLM en local sans GPU cloud.",
                metadata={"source": "test-doc-1", "filename": "intro_ollama.pdf", "page": 1},
            ),
            Document(
                page_content="Qdrant est une base vectorielle optimisée pour la recherche sémantique.",
                metadata={"source": "test-doc-2", "filename": "intro_qdrant.pdf", "page": 1},
            ),
            Document(
                page_content="Le RAG combine la récupération documentaire avec la génération de texte.",
                metadata={"source": "test-doc-3", "filename": "intro_rag.pdf", "page": 1},
            ),
        ]
        await store.add_documents(docs)
        print(f"Upsert de {len(docs)} documents OK")

        results = await store.similarity_search("Comment fonctionne le RAG ?", k=2)
        assert len(results) <= 2, "Maximum k résultats"
        print(f"Résultats ({len(results)}) :")
        for r in results:
            print(f"  - [{r.metadata.get('filename')}] {r.page_content[:60]}...")
        print("OK")

    async def test_delete_document() -> None:
        print("\n=== TEST delete_document() ===")
        await store.delete_document("test-doc-1")
        await asyncio.sleep(0.5)  # Qdrant indexe de façon asynchrone
        results = await store.similarity_search("Ollama LLM local", k=5)
        sources = [r.metadata.get("source") for r in results]
        assert "test-doc-1" not in sources, "Le document supprimé ne doit plus apparaître"
        print(f"Sources restantes : {sources}")
        print("OK")

    async def run_all() -> None:
        await test_init_collection()
        await test_add_and_search()
        await test_delete_document()
        # Nettoyage
        store._client.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value="test-doc-2"))])
            ),
        )
        store._client.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value="test-doc-3"))])
            ),
        )
        print("\nTous les tests sont passés.")

    asyncio.run(run_all())
