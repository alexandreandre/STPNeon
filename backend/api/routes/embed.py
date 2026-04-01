"""
POST /embed-document — équivalent Edge Function `embed-document`.
Génère un embedding (logique OpenAI identique à Deno) et insère dans `knowledge_documents`.
"""

import json
from typing import Any

import httpx
from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import settings
from core.pipeline_instance import get_pipeline

router = APIRouter()
pipeline = get_pipeline()


class EmbedBody(BaseModel):
    title: str
    content: str
    source_type: str | None = "manual"
    file_path: str | None = None


def _rest_headers_user(access_token: str) -> dict[str, str]:
    return {
        "apikey": settings.supabase_anon_key,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


async def _get_user_id(client: httpx.AsyncClient, access_token: str) -> str | None:
    r = await client.get(
        f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
        headers={
            "apikey": settings.supabase_anon_key,
            "Authorization": f"Bearer {access_token}",
        },
    )
    if r.status_code != 200:
        return None
    data = r.json()
    return data.get("id")


async def _generate_embedding(client: httpx.AsyncClient, title: str, content: str) -> list[float] | None:
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an embedding generator. Given the following document, return ONLY a JSON array "
                    "of exactly 768 floating point numbers between -1 and 1 that represent the semantic meaning "
                    "of the text. The numbers should capture the key concepts, topics, and meaning. "
                    "Return ONLY the JSON array, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Title: {title}\n\nContent: {content[:4000]}",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "store_embedding",
                    "description": "Store the semantic embedding vector for this document",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "embedding": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Array of exactly 768 floating point numbers representing the document embedding",
                            },
                        },
                        "required": ["embedding"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "store_embedding"}},
    }
    r = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        },
        json=body,
    )
    if not r.is_success:
        return None
    emb_data = r.json()
    try:
        tool_call = emb_data["choices"][0]["message"]["tool_calls"][0]
        args = json.loads(tool_call["function"]["arguments"])
        emb = args.get("embedding")
        if isinstance(emb, list):
            return emb
    except (KeyError, IndexError, json.JSONDecodeError, TypeError):
        pass
    return None


def _normalize_embedding(embedding: list[float]) -> list[float]:
    if len(embedding) == 768:
        return embedding
    if len(embedding) > 768:
        return embedding[:768]
    return embedding + [0.0] * (768 - len(embedding))


async def _insert_document(
    client: httpx.AsyncClient,
    access_token: str,
    row: dict[str, Any],
) -> tuple[int | None, str | None]:
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/knowledge_documents"
    r = await client.post(url, headers=_rest_headers_user(access_token), json=row)
    if r.status_code not in (200, 201):
        return None, r.text
    data = r.json()
    if isinstance(data, list) and data:
        return data[0].get("id"), None
    if isinstance(data, dict):
        return data.get("id"), None
    return None, "Réponse inattendue"


@router.post("/embed-document")
async def embed_document(
    body: EmbedBody,
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "No authorization header"})

    access_token = authorization.replace("Bearer ", "", 1).strip()
    if not body.title or not body.content:
        return JSONResponse(status_code=400, content={"error": "Titre et contenu requis"})

    async with httpx.AsyncClient(timeout=120.0) as client:
        user_id = await _get_user_id(client, access_token)
        if not user_id:
            return JSONResponse(status_code=401, content={"error": "Non authentifié"})

        embedding = await _generate_embedding(client, body.title, body.content)

        source_type = body.source_type or "manual"
        file_path = body.file_path

        if embedding is None:
            row = {
                "user_id": user_id,
                "title": body.title,
                "content": body.content,
                "source_type": source_type,
                "file_path": file_path,
            }
            doc_id, err = await _insert_document(client, access_token, row)
            if err:
                return JSONResponse(status_code=500, content={"error": err})
        else:
            emb = _normalize_embedding(embedding)
            row = {
                "user_id": user_id,
                "title": body.title,
                "content": body.content,
                "source_type": source_type,
                "file_path": file_path,
                "embedding": json.dumps(emb),
            }
            doc_id, err = await _insert_document(client, access_token, row)
            if err:
                return JSONResponse(status_code=500, content={"error": err})

        # À ce stade, le document est bien présent dans Supabase.
        # On le (ré)indexe immédiatement dans Qdrant pour que le RAG y ait accès.
        try:
            source_id = f"supabase:{doc_id}"
            # On supprime d'abord les anciens points éventuels pour ce document,
            # afin d'éviter les doublons et rendre l'opération idempotente.
            await pipeline._store.delete_document(source_id)  # type: ignore[attr-defined]
            await pipeline.ingest_document(
                text=body.content,
                metadata={
                    "source": source_id,
                    "filename": body.title or file_path or f"doc_{doc_id}",
                    "page": 1,
                    "source_type": source_type,
                    "file_path": file_path,
                },
            )
        except Exception:
            # On ne casse pas l'API si l'indexation Qdrant échoue : le document reste
            # disponible côté Supabase et pourra être resynchronisé plus tard.
            pass

        return {"id": str(doc_id), "embedded": embedding is not None}
