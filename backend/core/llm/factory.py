import os

from config import settings
from core.llm.anthropic import AnthropicProvider
from core.llm.base import BaseLLMProvider
from core.llm.gemini import GeminiProvider
from core.llm.mistral_api import MistralAPIProvider
from core.llm.ollama import OllamaProvider
from core.llm.openai import OpenAIProvider
from core.llm.openrouter import OpenRouterProvider
from core.llm.openwebui import OpenWebUIProvider, build_openwebui_chat_files

_REGISTRY: dict[str, type[BaseLLMProvider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
    "mistral-api": MistralAPIProvider,
    "openrouter": OpenRouterProvider,
    "openwebui": OpenWebUIProvider,
}

# Cache par (provider, model) — une instance par combinaison
_instances: dict[tuple[str, str], BaseLLMProvider] = {}


def get_llm_provider(
    provider: str | None = None,
    model: str | None = None,
) -> BaseLLMProvider:
    """
    Retourne une instance du provider demandé.

    Si provider est None → lit LLM_PROVIDER depuis os.environ (défaut: "ollama")
    Si model est None → utilise le modèle par défaut du provider

    Cache par (provider, model) — une instance par combinaison.
    """
    provider_name = provider or os.environ.get("LLM_PROVIDER", "ollama")

    if provider_name not in _REGISTRY:
        raise ValueError(
            f"Provider '{provider_name}' inconnu. "
            f"Disponibles : {list(_REGISTRY)}"
        )

    default_models: dict[str, str] = {
        "ollama": settings.ollama_llm_model,
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-haiku-20240307",
        "mistral-api": "mistral-small-latest",
        "openrouter": settings.openrouter_llm_model,
        "openwebui": settings.openwebui_model,
    }

    if provider_name == "openwebui":
        if not settings.openwebui_base_url.strip() or not settings.openwebui_api_key.strip():
            raise ValueError(
                "Open WebUI : renseigner OPENWEBUI_BASE_URL et OPENWEBUI_API_KEY côté backend."
            )
        if not settings.openwebui_model.strip():
            raise ValueError("Open WebUI : renseigner OPENWEBUI_MODEL (identifiant modèle côté instance).")
        resolved_model = settings.openwebui_model.strip()
    else:
        resolved_model = model or default_models.get(provider_name, "")

    cache_key = (provider_name, resolved_model)
    if cache_key in _instances:
        return _instances[cache_key]

    if provider_name == "ollama":
        instance: BaseLLMProvider = OllamaProvider(
            base_url=settings.ollama_base_url,
            model=resolved_model,
            timeout=settings.llm_timeout,
        )
    elif provider_name == "openai":
        instance = OpenAIProvider(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=resolved_model,
        )
    elif provider_name == "openrouter":
        instance = OpenRouterProvider(
            api_key=settings.openrouter_api_key,
            model=resolved_model,
            timeout=settings.llm_timeout,
        )
    elif provider_name == "openwebui":
        instance = OpenWebUIProvider(
            base_url=settings.openwebui_base_url.strip(),
            api_key=settings.openwebui_api_key.strip(),
            model=resolved_model,
            timeout=settings.llm_timeout,
            chat_files=build_openwebui_chat_files(),
        )
    elif provider_name == "gemini":
        instance = GeminiProvider(
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            model=resolved_model,
        )
    elif provider_name == "anthropic":
        instance = AnthropicProvider(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            model=resolved_model,
        )
    elif provider_name == "mistral-api":
        instance = MistralAPIProvider(
            api_key=os.environ.get("MISTRAL_API_KEY", ""),
            model=resolved_model,
        )

    _instances[cache_key] = instance
    return instance


def list_providers() -> dict[str, list[str]]:
    """
    Retourne tous les providers disponibles avec leurs modèles supportés.
    Utilisé par l'endpoint /api/llm/providers.
    """
    return {
        "ollama": [
            "mistral", "mistral:7b", "gemma:7b", "gemma:2b",
            "llama3:8b", "deepseek-r1", "phi-3",
        ],
        "openai": ["gpt-4o-mini", "gpt-4o"],
        "gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
        "anthropic": ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
        "mistral-api": ["mistral-small-latest", "mistral-medium-latest"],
    }
