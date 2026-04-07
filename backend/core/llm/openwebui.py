import json
from typing import Any, AsyncGenerator, TypedDict
from urllib.parse import urljoin

import httpx

from config import settings
from core.llm.base import BaseLLMProvider, LLMProviderError

_PROVIDER = "openwebui"


def build_openwebui_chat_files() -> list[dict[str, str]] | None:
    """
    Construit le tableau `files` pour POST /api/chat/completions (RAG Open WebUI).

    Voir https://docs.openwebui.com/reference/api-endpoints/ (section RAG).
    """
    raw = (settings.openwebui_chat_files_json or "").strip()
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, list) or len(data) == 0:
            return None
        out: list[dict[str, str]] = []
        for item in data:
            if isinstance(item, dict) and item.get("type") and item.get("id") is not None:
                out.append({"type": str(item["type"]), "id": str(item["id"])})
        return out or None

    cid = (settings.openwebui_knowledge_collection_id or "").strip()
    if cid:
        return [{"type": "collection", "id": cid}]
    return None


class OpenWebUIUsage(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class OpenWebUIProvider(BaseLLMProvider):
    """Client pour l'API Open WebUI (format compatible OpenAI, Bearer API key)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        chat_files: list[dict[str, str]] | None = None,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self.chat_files = chat_files
        base = self._base_url.rstrip("/") + "/"
        path = (settings.openwebui_chat_path or "/api/chat/completions").lstrip("/")
        self._url = urljoin(base, path)

    def _httpx_timeout(self) -> httpx.Timeout:
        """Lecture jusqu'à `self.timeout` s (premier jeton lent ou réponse complète) ; connexion bornée."""
        t = float(self.timeout)
        return httpx.Timeout(connect=min(30.0, t), read=t, write=min(120.0, t), pool=10.0)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages: list[dict], *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream}
        if self.chat_files:
            payload["files"] = self.chat_files
        return payload

    async def generate(self, messages: list[dict]) -> str:
        payload = self._payload(messages, stream=False)
        try:
            async with httpx.AsyncClient(timeout=self._httpx_timeout()) as client:
                resp = await client.post(self._url, headers=self._headers(), json=payload)
        except httpx.ConnectError as exc:
            raise LLMProviderError(
                _PROVIDER, "Open WebUI inaccessible — vérifier OPENWEBUI_BASE_URL et le réseau"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMProviderError(_PROVIDER, f"Timeout après {self.timeout}s") from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(_PROVIDER, f"Erreur réseau : {exc}") from exc

        if resp.status_code != 200:
            raise LLMProviderError(_PROVIDER, resp.text, status_code=resp.status_code)

        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    async def stream(
        self, messages: list[dict]
    ) -> AsyncGenerator[tuple[str | None, OpenWebUIUsage | None], None]:
        payload = self._payload(messages, stream=True)
        try:
            async with httpx.AsyncClient(timeout=self._httpx_timeout()) as client:
                async with client.stream(
                    "POST", self._url, headers=self._headers(), json=payload
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise LLMProviderError(
                            _PROVIDER, body.decode(errors="replace"),
                            status_code=resp.status_code,
                        )
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[len("data: ") :]
                        if raw == "[DONE]":
                            break
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        usage: OpenWebUIUsage | None = chunk.get("usage")  # type: ignore[assignment]
                        content = ""
                        try:
                            content = chunk["choices"][0]["delta"].get("content", "") or ""
                        except (KeyError, IndexError, TypeError):
                            content = ""

                        if content or usage:
                            yield content or None, usage
        except LLMProviderError:
            raise
        except httpx.ConnectError as exc:
            raise LLMProviderError(
                _PROVIDER, "Open WebUI inaccessible — vérifier OPENWEBUI_BASE_URL et le réseau"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMProviderError(_PROVIDER, f"Timeout après {self.timeout}s") from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(_PROVIDER, f"Erreur réseau : {exc}") from exc
