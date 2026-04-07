"""
Router FastAPI pour les endpoints LLM.
GET /api/llm/providers               : liste tous les providers et leurs modèles (legacy)
GET /api/llm/providers/current       : provider et modèle actuellement actifs
GET /api/llm/openrouter/models       : liste des modèles OpenRouter (pour le sélecteur UI)
"""

import os

import httpx
from fastapi import APIRouter, HTTPException

from config import settings
from core.llm import get_llm_provider, list_providers
from core.llm_stats import build_usage_aggregates

router = APIRouter(prefix="/llm")

# ID stable côté STPNeon (sélecteur assistant + comparateur + stats) — le nom réel du modèle Open WebUI reste dans OPENWEBUI_MODEL.
STPNEON_OPENWEBUI_CATALOG_ID = "stpneon/openwebui"


def _stpneon_openwebui_configured() -> bool:
    return bool(
        (settings.openwebui_api_key or "").strip()
        and (settings.openwebui_base_url or "").strip()
        and (settings.openwebui_model or "").strip()
    )


def _stpneon_openwebui_selector_model() -> dict:
    """Entrée format /openrouter/models pour le sélecteur UI."""
    return {
        "id": STPNEON_OPENWEBUI_CATALOG_ID,
        "name": "STPNeon OpenWebUI",
        "description": "Instance Open WebUI configurée côté infrastructure (API externe).",
        "context_length": 0,
        "pricing": {},
        "license_kind": "proprietary",
        "pricing_per_1m_usd": {"input": None, "output": None},
    }


def _stpneon_openwebui_comparator_catalog_entry() -> dict:
    """Entrée format /comparator model_catalog."""
    return {
        "id": STPNEON_OPENWEBUI_CATALOG_ID,
        "name": "STPNeon OpenWebUI",
        "open_weights_category": "unknown",
        "context_length": None,
        "context_meta": None,
        "pricing": {
            "raw": {},
            "prompt_per_1m_usd": None,
            "completion_per_1m_usd": None,
            "input_per_1m_usd": None,
            "output_per_1m_usd": None,
        },
        "local_hardware_hint": None,
        "top_provider": "openwebui",
    }


@router.get("/providers")
async def get_providers():
    """Retourne tous les providers disponibles avec leurs modèles (legacy)."""
    return list_providers()


@router.get("/providers/current")
async def get_current_provider():
    """Retourne le provider et modèle actuellement actifs."""
    provider = os.environ.get("LLM_PROVIDER", "ollama")
    default_models = {
        "ollama": settings.ollama_llm_model,
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-haiku-20240307",
        "mistral-api": "mistral-small-latest",
        "openrouter": settings.openrouter_llm_model,
    }
    return {"provider": provider, "model": default_models.get(provider, "")}


def _build_openrouter_headers() -> dict[str, str]:
    if not settings.openrouter_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY non configurée côté backend.",
        )

    headers: dict[str, str] = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
    }
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_app_title:
        headers["X-OpenRouter-Title"] = settings.openrouter_app_title
    return headers


def _classify_open_weights(model_id: str | None) -> str:
    """
    Classe très grossièrement un modèle OpenRouter en trois catégories :
    - "open"    : modèles dont les poids sont publiés (familles Llama, Mistral OSS, Gemma, Qwen, Falcon, etc.)
    - "closed"  : API propriétaires bien connues (OpenAI, Anthropic, Gemini, Grok, Cohere, Databricks, etc.)
    - "unknown" : tout le reste (par défaut)

    Il n'y a pas aujourd'hui de champ officiel dans l'API OpenRouter pour cela ; on s'appuie donc
    sur des conventions de nommage d'ID, que l'on pourra faire évoluer au fil du temps.
    """
    if not model_id:
        return "unknown"

    mid = model_id.lower()

    open_prefixes = [
        # Meta Llama 3 / 3.1 / 3.2 / 3.3 / Guard (open weights)
        "meta-llama/llama-3",
        "meta-llama/llama-guard",
        "meta-llama/llama-3.1",
        "meta-llama/llama-3.2",
        "meta-llama/llama-3.3",
        # Meta Llama 4 famille (poids annoncés comme ouverts)
        "meta-llama/llama-4",
        # Community / HF fine-tunes sur Llama, Mistral, etc.
        "alfredpros/",
        "alpindale/",
        "anthracite-org/",
        "cognitivecomputations/",
        "eleutherai/",
        "gryphe/",
        "sao10k/",
        "thedrummer/",
        "tngtech/",
        "undi95/",
        "aion-labs/aion-rp-llama",
        # Mistral open-weight familles (id « mistral/... » sur OpenRouter)
        "mistral/mistral-7b",
        "mistral/mistral-small",
        "mistral/mistral-nemo",
        "mistral/open-mistral",
        "mistral/smollm",
        # mistralai/mistral-small-3.x : offres API Mistral (non open-weight) — classées via le préfixe mistralai/ (closed)
        "mistral/mixtral-8x7b",
        "mistral/codestral",
        # Même famille open-weight sous le vendor « mistralai/ » (7B, Nemo, Ministral…)
        # — doit précéder la règle fermée « mistralai/ » ci-dessous (open testé en premier).
        "mistralai/mistral-7b",
        "mistralai/mistral-nemo",
        "mistralai/open-mistral-7b",
        "mistralai/ministral-3b",
        "mistralai/ministral-8b",
        # Qwen open weights
        "qwen/qwen",
        "qwen2",
        "qwen2.5",
        "qwen3",
        "qwen3.5",
        "qwen3.6",
        "qwen/qwq",
        # Gemma
        "google/gemma",
        # Falcon
        "tiiuae/falcon",
        # DeepSeek
        "deepseek/",
        "nex-agi/deepseek",
        # NousResearch fine-tunes (sur base open-weight)
        "nousresearch/",
        # Divers OSS connus
        "prime-intellect/",
        # Hugging Face proxys
        "huggingface/",
        # AllenAI OLMo familles (weights ouverts)
        "allenai/olmo",
    ]

    closed_prefixes = [
        # GAFAM / grands clouds & API propriétaires
        "openai/",
        "gpt-",
        "anthropic/",
        "claude-",
        "google/gemini",
        "google/lyria",
        "amazon/",
        "bedrock/",
        "ibm-granite/",
        "microsoft/",
        "azure/",
        "nvidia/",
        "moonshotai/",
        "minimax/",
        "reka/",
        "rekaai/",
        "stepfun/",
        "tencent/",
        "x-ai/",
        "cohere/",
        "databricks/",
        "perplexity/",
        "bytedance",
        "bytedance-seed/",
        "baidu/",
        "ai21/",
        "inflection/",
        "upstage/",
        "writer/",
        "xiaomi/",
        "z-ai/",
        # Mistral API-only (hors préfixes open-weight mistralai/… listés plus haut)
        "mistralai/",
        # Divers vendors / routers supplémentaires
        "aion-labs/",
        "arcee-ai/",
        "deepcogito/",
        "essentialai/",
        "inception/",
        "kwaipilot/",
        "liquid/",
        "mancer/",
        "meituan/",
        "morph/",
        "relace/",
        "switchpoint/",
        "alibaba/",
        "openrouter/",
    ]

    if any(mid.startswith(p) for p in open_prefixes):
        return "open"
    if any(mid.startswith(p) for p in closed_prefixes):
        return "closed"
    return "unknown"


def _build_local_hardware_hint(model_id: str | None, category: str) -> dict | None:
    """
    Retourne un petit objet d'aide à la lecture pour la colonne « Ressources locales (≈30 pers.) ».

    On utilise des ordres de grandeur classiques par taille de modèle (B de paramètres) en Q4_K_M,
    suffisants pour se situer (et non pour dimensionner précisément une infra de prod).
    """
    if not model_id:
        return None

    mid = model_id.lower()

    # On ne renseigne une estimation que pour les modèles identifiés open‑source.
    if category != "open":
        return None

    # Extraction très simple de la taille en "b" (7b, 8b, 14b, 32b, 70b, 120b, 200b, 400b…).
    size_gb_q4: float | None = None
    notes: list[str] = []

    if "1.2b" in mid or "1.3b" in mid:
        size_gb_q4 = 4.0
    elif "2b" in mid or "3b" in mid:
        size_gb_q4 = 6.0
    elif "6b" in mid or "7b" in mid or "8b" in mid or "9b" in mid:
        size_gb_q4 = 8.0
    elif "12b" in mid or "13b" in mid or "14b" in mid:
        size_gb_q4 = 16.0
    elif "22b" in mid or "24b" in mid or "27b" in mid or "30b" in mid or "32b" in mid or "35b" in mid:
        size_gb_q4 = 24.0
    elif "70b" in mid or "72b" in mid:
        size_gb_q4 = 48.0
    elif "120b" in mid or "122b" in mid:
        size_gb_q4 = 80.0
    elif "200b" in mid or "235b" in mid or "253b" in mid or "260b" in mid or "300b" in mid or "397b" in mid:
        size_gb_q4 = 120.0
    elif "400b" in mid or "480b" in mid or "671b" in mid:
        size_gb_q4 = 160.0

    if size_gb_q4 is None:
        # Fallback très grossier : on ne met rien plutôt qu'une valeur fantaisiste.
        notes.append("Taille GPU indicative non déterminée automatiquement (à compléter manuellement si besoin).")
        return {
            "matched_id": model_id,
            "gpu_notes": " / ".join(notes) if notes else "",
        }

    notes.append(
        "Estimation Q4_K_M basée sur la taille du modèle (ordre de grandeur, non contractuel).",
    )

    return {
        "matched_id": model_id,
        "vram_gb_q4_k_m_typical": size_gb_q4,
        "gpu_notes": " ".join(notes),
    }


def _normalize_pricing(raw: dict | None) -> dict:
    """
    Normalise le bloc pricing OpenRouter en champs numériques explicites:
    - prompt_per_1m_usd / completion_per_1m_usd
    - input_per_1m_usd / output_per_1m_usd
    """
    data = raw or {}

    def _to_float(v: object) -> float | None:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return None
        return None

    # OpenRouter renvoie des prix « par token » (USD/token).
    # Dans l’UI on affiche « $ / 1M tokens » : on multiplie donc systématiquement par 1e6.
    def _scale_1m(v: float | None) -> float | None:
        return v * 1_000_000 if v is not None else None

    prompt = _scale_1m(_to_float(data.get("prompt")))
    completion = _scale_1m(_to_float(data.get("completion")))
    # Si OpenRouter expose input/output explicitement, on les prend ; sinon fallback sur prompt/completion.
    input_price = _scale_1m(_to_float(data.get("input"))) or prompt
    output_price = _scale_1m(_to_float(data.get("output"))) or completion

    # Certaines entrées OpenRouter utilisent des sentinelles négatives (p. ex. -1000000) pour les routeurs.
    # On les considère comme « non renseignées ».
    for key in ("prompt_per_1m_usd", "completion_per_1m_usd", "input_per_1m_usd", "output_per_1m_usd"):
        pass

    return {
        "raw": data,
        "prompt_per_1m_usd": prompt,
        "completion_per_1m_usd": completion,
        "input_per_1m_usd": input_price,
        "output_per_1m_usd": output_price,
    }


# Ordre d'affichage prioritaire dans le sélecteur (pertinence RAG + français / doc interne).
# Préfixes d'ID OpenRouter ; les plus spécifiques doivent précéder les plus génériques.
_CURATED_MODEL_ORDER: list[str] = [
    "stpneon/openwebui",
    "openai/gpt-4o-mini",
    "openai/gpt-4o-mini-search-preview",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3.1-flash-lite",
    "google/gemini-2.5-flash-lite",
    "mistralai/mistral-small-3.2-24b-instruct",
    "mistralai/mistral-small-3.2-24b",
    "mistralai/mistral-small-3.2",
    "meta-llama/llama-3.1-8b",
    "mistral/mistral-nemo",
    "mistral/mistral-7b",
    "mistral/open-mistral-7b",
    "mistralai/mistral-nemo",
    "mistralai/mistral-7b",
    "mistralai/open-mistral-7b",
    "qwen/qwen-2.5-7b",
    "qwen/qwen-2.5-14b",
    "qwen/qwen2.5-14b",
    "qwen/qwen2.5-7b",
    "google/gemma-2-27b",
    "google/gemma-2-9b",
]


def _curated_model_rank(model_id: str | None) -> int:
    if not model_id:
        return 10_000
    mid = model_id.lower()
    for i, prefix in enumerate(_CURATED_MODEL_ORDER):
        if mid.startswith(prefix.lower()):
            return i
    return 10_000


def _license_kind_for_ui(model_id: str | None) -> str:
    """open_weights | proprietary | unknown — aligné sur les libellés du sélecteur."""
    c = _classify_open_weights(model_id)
    if c == "open":
        return "open_weights"
    if c == "closed":
        return "proprietary"
    return "unknown"


def _openrouter_model_sort_key(entry: dict) -> tuple:
    """API propriétaire d'abord, puis poids ouverts, puis autres ; modèles « curated » en tête de chaque bloc."""
    lk = entry.get("license_kind") or "unknown"
    grp = 0 if lk == "proprietary" else 1 if lk == "open_weights" else 2
    mid = entry.get("id") or ""
    name = (entry.get("name") or mid).lower()
    return (grp, _curated_model_rank(mid), name)


def _is_free_or_router_model(m: dict) -> bool:
    """
    Filtre les modèles que l'on ne souhaite pas exposer au client :
    - routeurs internes OpenRouter (openrouter/auto, openrouter/free, …)
    - variantes explicitement gratuites (:free)
    - modèles dont tous les prix connus sont à 0
    """
    mid = str(m.get("id") or "").lower()
    if mid.startswith("openrouter/"):
        return True
    if mid.endswith(":free"):
        return True

    pricing = m.get("pricing") or {}

    def _to_float_local(v: object) -> float | None:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return None
        return None

    vals = [
        _to_float_local(pricing.get("input")),
        _to_float_local(pricing.get("output")),
        _to_float_local(pricing.get("prompt")),
        _to_float_local(pricing.get("completion")),
    ]
    non_null = [v for v in vals if v is not None]
    if non_null and all(v == 0 for v in non_null):
        return True
    return False

async def _fetch_openrouter_models() -> list[dict]:
    headers = _build_openrouter_headers()

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Erreur réseau vers OpenRouter : {exc}") from exc

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Erreur OpenRouter models : {resp.text}",
        )

    payload = resp.json()
    return payload.get("data", [])


@router.get("/openrouter/models")
async def get_openrouter_models():
    """
    Retourne la liste des modèles OpenRouter disponibles pour le chat.
    Utilisé par le sélecteur de modèle dans l'UI.
    """
    raw_models = await _fetch_openrouter_models()
    # On retire les routeurs internes et modèles gratuits pour ne proposer que des modèles pertinents.
    raw_models = [m for m in raw_models if not _is_free_or_router_model(m)]

    models: list[dict] = []
    for m in raw_models:
        mid = m.get("id")
        norm = _normalize_pricing(m.get("pricing"))
        lk = _license_kind_for_ui(mid)
        models.append(
            {
                "id": mid,
                "name": m.get("name") or mid,
                "description": m.get("description") or "",
                "context_length": m.get("context_length", 0),
                "pricing": m.get("pricing", {}),
                "license_kind": lk,
                "pricing_per_1m_usd": {
                    "input": norm["input_per_1m_usd"],
                    "output": norm["output_per_1m_usd"],
                },
            }
        )

    models.sort(key=_openrouter_model_sort_key)

    if _stpneon_openwebui_configured():
        models.insert(0, _stpneon_openwebui_selector_model())

    return {
        "default_model": settings.openrouter_llm_model,
        "models": models,
    }


@router.get("/comparator")
async def get_llm_comparator():
    """
    Endpoint utilisé par la page LLMComparator du frontend.

    Pour l'instant on renvoie :
    - des stats d'usage et de feedback vides
    - un catalogue de modèles basé sur les modèles OpenRouter
    - des agrégats globaux à zéro
    Cela permet au frontend d'afficher la page sans erreur même si aucun tracking
    détaillé n'est encore branché.
    """
    raw_models = await _fetch_openrouter_models()
    # Même filtrage que pour /openrouter/models : pas de modèles « free » ni de routeurs internes.
    raw_models = [m for m in raw_models if not _is_free_or_router_model(m)]

    model_catalog: list[dict] = []
    if _stpneon_openwebui_configured():
        model_catalog.append(_stpneon_openwebui_comparator_catalog_entry())

    for m in raw_models:
        mid = m.get("id")
        category = _classify_open_weights(mid)
        model_catalog.append(
            {
                "id": mid,
                "name": m.get("name") or mid,
                # Catégorie « open / closed / unknown » — heuristique maison.
                "open_weights_category": category,
                # Fenêtre de contexte telle que renvoyée par OpenRouter.
                "context_length": m.get("context_length"),
                # Métadonnées additionnelles (non renseignées pour l'instant).
                "context_meta": None,
                # Tarifs par million de tokens (normalisés pour le frontend).
                "pricing": _normalize_pricing(m.get("pricing")),
                # Indications matériel local (ordre de grandeur pour modèles open‑source).
                "local_hardware_hint": _build_local_hardware_hint(mid, category),
                # Provider / routeur éventuel côté OpenRouter (si disponible).
                "top_provider": m.get("top_provider"),
            }
        )

    usage_rows, global_stats, feedback_stats = build_usage_aggregates(model_catalog)

    payload = {
        "usage_rows": usage_rows,
        "model_catalog": model_catalog,
        "feedback_stats": feedback_stats,
        "global": global_stats,
        # Pas encore de documentation détaillée centralisée pour l'inférence locale.
        "local_hardware_documentation": [],
    }

    return payload
