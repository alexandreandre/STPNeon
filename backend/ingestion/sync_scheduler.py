"""
Planificateur de synchronisation SharePoint.
Exécute périodiquement (APScheduler AsyncIOScheduler) la détection des fichiers
nouveaux ou modifiés dans SharePoint, déclenche leur parsing et leur
indexation dans Qdrant, et journalise les résultats.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

if TYPE_CHECKING:
    from core.rag_pipeline import RAGPipeline
    from ingestion.sharepoint import SharePointConnector

logger = logging.getLogger(__name__)

_SYNC_INTERVAL_MINUTES = 30


# ---------------------------------------------------------------------------
# Tâches de synchronisation
# ---------------------------------------------------------------------------

async def _sync_changed_files(
    pipeline: "RAGPipeline",
    sharepoint: "SharePointConnector",
) -> None:
    """
    Job périodique (toutes les 30 min) : synchronise les fichiers modifiés.

    Étapes :
      1. Récupère les fichiers modifiés depuis le dernier delta SharePoint
      2. Supprime les anciens points Qdrant (par source_id)
      3. Télécharge et parse le nouveau contenu
      4. Ré-indexe dans Qdrant via RAGPipeline.ingest_document()

    En cas d'erreur sur un fichier individuel, loggue et continue.
    """
    from ingestion.file_parser import parse_file

    logger.info("Sync démarrée (job=changed_files).")
    t_start = time.monotonic()

    try:
        changed_files = await sharepoint.get_changed_files()
    except Exception as exc:
        logger.error("Impossible de récupérer les fichiers modifiés SharePoint : %s", exc)
        return

    if not changed_files:
        logger.info("Aucun fichier modifié détecté. Sync terminée en %.1fs.", time.monotonic() - t_start)
        return

    logger.info("%d fichier(s) modifié(s) détecté(s).", len(changed_files))

    indexed_count = 0
    error_count = 0

    for file_info in changed_files:
        source_id: str = file_info.get("id", file_info.get("name", ""))
        filename: str = file_info.get("name", source_id)
        local_path: str | None = file_info.get("local_path")

        if not local_path:
            logger.warning("Pas de chemin local pour '%s', téléchargement nécessaire.", filename)
            try:
                local_path = await sharepoint.download_file(file_info)
            except Exception as exc:
                logger.error("Téléchargement échoué pour '%s' : %s", filename, exc)
                error_count += 1
                continue

        # Supprime les points existants pour ce document
        try:
            await pipeline._store.delete_document(source_id)
            logger.debug("Anciens points supprimés pour source='%s'.", source_id)
        except Exception as exc:
            logger.warning("Suppression partielle pour '%s' : %s", source_id, exc)

        # Parse le fichier local
        docs = parse_file(local_path)
        if not docs:
            logger.warning("Aucun contenu extrait de '%s', ignoré.", filename)
            continue

        # Ré-indexe dans Qdrant
        try:
            chunk_count = 0
            for doc in docs:
                doc.metadata.setdefault("source", source_id)
                n = await pipeline.ingest_document(
                    text=doc.page_content,
                    metadata=doc.metadata,
                )
                chunk_count += n
            indexed_count += 1
            logger.info("'%s' indexé — %d chunk(s).", filename, chunk_count)
        except Exception as exc:
            logger.error("Indexation échouée pour '%s' : %s", filename, exc)
            error_count += 1

    elapsed = time.monotonic() - t_start
    logger.info(
        "Sync terminée en %.1fs — %d fichier(s) indexé(s), %d erreur(s).",
        elapsed,
        indexed_count,
        error_count,
    )


async def _sync_full(
    pipeline: "RAGPipeline",
    sharepoint: "SharePointConnector",
) -> None:
    """
    Sync complète : récupère tous les fichiers du drive SharePoint et les indexe.
    Appelée au démarrage uniquement si la collection Qdrant est vide.
    """
    from ingestion.file_parser import parse_file

    logger.info("Sync complète démarrée (collection Qdrant vide au démarrage).")
    t_start = time.monotonic()

    try:
        all_files = await sharepoint.list_all_files()
    except Exception as exc:
        logger.error("Impossible de lister les fichiers SharePoint : %s", exc)
        return

    logger.info("%d fichier(s) total(aux) à indexer.", len(all_files))

    indexed_count = 0
    error_count = 0

    for file_info in all_files:
        source_id: str = file_info.get("id", file_info.get("name", ""))
        filename: str = file_info.get("name", source_id)
        local_path: str | None = file_info.get("local_path")

        if not local_path:
            try:
                local_path = await sharepoint.download_file(file_info)
            except Exception as exc:
                logger.error("Téléchargement échoué pour '%s' : %s", filename, exc)
                error_count += 1
                continue

        docs = parse_file(local_path)
        if not docs:
            logger.warning("Aucun contenu extrait de '%s', ignoré.", filename)
            continue

        try:
            chunk_count = 0
            for doc in docs:
                doc.metadata.setdefault("source", source_id)
                n = await pipeline.ingest_document(
                    text=doc.page_content,
                    metadata=doc.metadata,
                )
                chunk_count += n
            indexed_count += 1
            logger.info("'%s' indexé — %d chunk(s).", filename, chunk_count)
        except Exception as exc:
            logger.error("Indexation échouée pour '%s' : %s", filename, exc)
            error_count += 1

    elapsed = time.monotonic() - t_start
    logger.info(
        "Sync complète terminée en %.1fs — %d/%d fichier(s) indexé(s), %d erreur(s).",
        elapsed,
        indexed_count,
        len(all_files),
        error_count,
    )


async def _collection_is_empty(pipeline: "RAGPipeline") -> bool:
    """
    Retourne True si la collection Qdrant ne contient aucun point.
    Utilisé au démarrage pour décider si une sync complète est nécessaire.
    """
    try:
        info = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pipeline._store._client.get_collection(
                pipeline._store._collection
            ),
        )
        count: int = info.points_count or 0
        logger.debug("Collection '%s' — %d point(s).", pipeline._store._collection, count)
        return count == 0
    except Exception as exc:
        logger.warning("Impossible de vérifier le nombre de points Qdrant : %s", exc)
        return False


# ---------------------------------------------------------------------------
# Scheduler public
# ---------------------------------------------------------------------------

class SyncScheduler:
    """
    Planificateur de synchronisation SharePoint ↔ Qdrant.

    Enregistre deux jobs APScheduler :
      - startup_sync  : sync complète si la collection est vide au démarrage
      - periodic_sync : sync des fichiers modifiés toutes les 30 minutes

    Usage :
        scheduler = SyncScheduler(pipeline, sharepoint_connector)
        scheduler.start()          # à appeler dans le lifespan FastAPI
        ...
        scheduler.stop()           # à appeler à l'arrêt de l'app
    """

    def __init__(
        self,
        pipeline: "RAGPipeline",
        sharepoint: "SharePointConnector",
    ) -> None:
        self._pipeline = pipeline
        self._sharepoint = sharepoint
        self._scheduler = AsyncIOScheduler(timezone="UTC")

    def start(self) -> None:
        """
        Démarre le scheduler et enregistre les jobs.

        - Job immédiat (delay=5s) : sync complète si collection vide
        - Job périodique : sync des fichiers modifiés toutes les 30 minutes
        """
        # Job de startup : déclenché 5s après le démarrage
        self._scheduler.add_job(
            self._run_startup_sync,
            trigger="date",
            run_date=None,  # APScheduler calcule "maintenant + next_run_time" via delay
            id="startup_sync",
            name="Sync complète au démarrage",
            misfire_grace_time=60,
        )

        # Job périodique toutes les 30 minutes
        self._scheduler.add_job(
            self._run_changed_sync,
            trigger=IntervalTrigger(minutes=_SYNC_INTERVAL_MINUTES),
            id="periodic_sync",
            name=f"Sync fichiers modifiés (/{_SYNC_INTERVAL_MINUTES}min)",
            replace_existing=True,
            misfire_grace_time=120,
        )

        self._scheduler.start()
        logger.info(
            "SyncScheduler démarré — sync complète au boot, sync delta toutes les %dmin.",
            _SYNC_INTERVAL_MINUTES,
        )

    def stop(self) -> None:
        """Arrête proprement le scheduler (à appeler dans le shutdown FastAPI)."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("SyncScheduler arrêté.")

    async def _run_startup_sync(self) -> None:
        """Exécute la sync complète si la collection Qdrant est vide."""
        if await _collection_is_empty(self._pipeline):
            logger.info("Collection vide détectée — lancement de la sync complète.")
            await _sync_full(self._pipeline, self._sharepoint)
        else:
            logger.info(
                "Collection non vide au démarrage — sync complète ignorée. "
                "La sync delta prendra le relais."
            )

    async def _run_changed_sync(self) -> None:
        """Exécute la sync des fichiers modifiés."""
        await _sync_changed_files(self._pipeline, self._sharepoint)

    def trigger_full_sync_now(self) -> None:
        """
        Déclenche immédiatement une sync complète (hors planning).
        Utile pour un endpoint d'administration ou un test manuel.
        """
        self._scheduler.add_job(
            _sync_full,
            args=[self._pipeline, self._sharepoint],
            id="manual_full_sync",
            name="Sync complète manuelle",
            replace_existing=True,
        )
        logger.info("Sync complète manuelle planifiée pour exécution immédiate.")

    def trigger_delta_sync_now(self) -> None:
        """
        Déclenche immédiatement une sync delta (hors planning).
        Utile pour un test ou un rafraîchissement forcé depuis l'UI admin.
        """
        self._scheduler.add_job(
            _sync_changed_files,
            args=[self._pipeline, self._sharepoint],
            id="manual_delta_sync",
            name="Sync delta manuelle",
            replace_existing=True,
        )
        logger.info("Sync delta manuelle planifiée pour exécution immédiate.")


# ---------------------------------------------------------------------------
# Tests basiques — exécuter avec : python ingestion/sync_scheduler.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys
    import tempfile
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # ---------------------------------------------------------------------------
    # Mock SharePointConnector
    # ---------------------------------------------------------------------------
    class MockSharePoint:
        """Simule un connecteur SharePoint avec des fichiers TXT locaux."""

        def __init__(self, tmp_dir: str) -> None:
            self._dir = tmp_dir
            self._delta_call_count = 0

        async def get_changed_files(self) -> list[dict]:
            self._delta_call_count += 1
            if self._delta_call_count == 1:
                # Premier appel : 2 fichiers modifiés
                return [
                    {
                        "id": "file-rh-001",
                        "name": "politique_rh.txt",
                        "local_path": str(Path(self._dir) / "politique_rh.txt"),
                    },
                    {
                        "id": "file-tech-001",
                        "name": "guide_technique.txt",
                        "local_path": str(Path(self._dir) / "guide_technique.txt"),
                    },
                ]
            # Appels suivants : aucun changement
            return []

        async def list_all_files(self) -> list[dict]:
            return [
                {
                    "id": "file-intro-001",
                    "name": "introduction.txt",
                    "local_path": str(Path(self._dir) / "introduction.txt"),
                }
            ]

        async def download_file(self, file_info: dict) -> str:
            return file_info.get("local_path", "")

    # ---------------------------------------------------------------------------
    # Mock RAGPipeline
    # ---------------------------------------------------------------------------
    class MockPipeline:
        """Simule un RAGPipeline pour les tests sans Qdrant ni Ollama."""

        def __init__(self) -> None:
            self._ingested: list[dict] = []
            self._deleted: list[str] = []
            self._store = MagicMock()
            self._store._collection = "stpneon_test"
            self._store._client = MagicMock()
            # Simule une collection vide (points_count=0)
            mock_info = MagicMock()
            mock_info.points_count = 0
            self._store._client.get_collection.return_value = mock_info

        async def ingest_document(self, text: str, metadata: dict) -> int:
            self._ingested.append({"text_len": len(text), "metadata": metadata})
            return 1  # 1 chunk simulé

    async def test_sync_changed_files() -> None:
        print("\n=== TEST _sync_changed_files() ===")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Crée des fichiers TXT locaux
            Path(tmpdir, "politique_rh.txt").write_text(
                "Politique de congés : 25 jours ouvrés par an.", encoding="utf-8"
            )
            Path(tmpdir, "guide_technique.txt").write_text(
                "Guide d'installation Ollama et Qdrant.", encoding="utf-8"
            )

            sp = MockSharePoint(tmpdir)
            pipeline = MockPipeline()
            pipeline._store.delete_document = AsyncMock()

            await _sync_changed_files(pipeline, sp)

            assert pipeline._store.delete_document.call_count == 2, (
                f"Attendu 2 suppressions, obtenu {pipeline._store.delete_document.call_count}"
            )
            assert len(pipeline._ingested) == 2, (
                f"Attendu 2 ingestions, obtenu {len(pipeline._ingested)}"
            )
            print(f"  Fichiers indexés : {len(pipeline._ingested)}")
            print(f"  Suppressions Qdrant : {pipeline._store.delete_document.call_count}")
        print("OK")

    async def test_sync_changed_files_no_changes() -> None:
        print("\n=== TEST _sync_changed_files() — aucun changement ===")
        with tempfile.TemporaryDirectory() as tmpdir:
            sp = MockSharePoint(tmpdir)
            sp._delta_call_count = 99  # Simule qu'on est après le premier appel
            pipeline = MockPipeline()
            pipeline._store.delete_document = AsyncMock()

            await _sync_changed_files(pipeline, sp)

            assert len(pipeline._ingested) == 0
            print("OK — aucun fichier indexé comme attendu")

    async def test_sync_full() -> None:
        print("\n=== TEST _sync_full() ===")
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "introduction.txt").write_text(
                "Introduction à STPNeon, plateforme documentaire IA.", encoding="utf-8"
            )

            sp = MockSharePoint(tmpdir)
            pipeline = MockPipeline()

            await _sync_full(pipeline, sp)

            assert len(pipeline._ingested) == 1, (
                f"Attendu 1 ingestion, obtenu {len(pipeline._ingested)}"
            )
            print(f"  Fichiers indexés : {len(pipeline._ingested)}")
        print("OK")

    async def test_collection_is_empty_true() -> None:
        print("\n=== TEST _collection_is_empty() — collection vide ===")
        pipeline = MockPipeline()
        result = await _collection_is_empty(pipeline)
        assert result is True, "La collection simulée devrait être vide"
        print("OK")

    async def test_collection_is_empty_false() -> None:
        print("\n=== TEST _collection_is_empty() — collection non vide ===")
        pipeline = MockPipeline()
        mock_info = MagicMock()
        mock_info.points_count = 42
        pipeline._store._client.get_collection.return_value = mock_info
        result = await _collection_is_empty(pipeline)
        assert result is False, "La collection devrait être non vide"
        print("OK")

    async def test_scheduler_start_stop() -> None:
        print("\n=== TEST SyncScheduler.start() / stop() ===")
        with tempfile.TemporaryDirectory() as tmpdir:
            sp = MockSharePoint(tmpdir)
            pipeline = MockPipeline()
            scheduler = SyncScheduler(pipeline, sp)
            scheduler.start()
            assert scheduler._scheduler.running, "Le scheduler doit être en cours d'exécution"
            jobs = scheduler._scheduler.get_jobs()
            job_ids = [j.id for j in jobs]
            assert "periodic_sync" in job_ids, f"Job 'periodic_sync' manquant, jobs={job_ids}"
            print(f"  Jobs enregistrés : {job_ids}")
            scheduler.stop()
            print("OK")

    async def run_all() -> None:
        await test_sync_changed_files()
        await test_sync_changed_files_no_changes()
        await test_sync_full()
        await test_collection_is_empty_true()
        await test_collection_is_empty_false()
        await test_scheduler_start_stop()
        print("\nTous les tests sont passés.")

    asyncio.run(run_all())
