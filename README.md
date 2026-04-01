# Telko — Assistant documentaire interne

Plateforme d'assistance IA pour les employés d'une société de télécom.
Les utilisateurs interrogent en langage naturel la base documentaire interne, et reçoivent une réponse en français, argumentée et sourcée.

---

## Vue d'ensemble fonctionnelle

1. **Un collaborateur ouvre l'interface web Telko** (frontend React).
2. **Il pose une question** en français sur un document, un process interne, une politique RH, etc.
3. **Le backend FastAPI reçoit la requête** et :
   - récupère le **contexte documentaire** : documents pertinents issus de Supabase (base actuelle) et/ou de Qdrant (RAG).
4. **Un modèle LLM hébergé via OpenRouter** génère la réponse :
   - en tenant compte du contexte documentaire ;
   - en respectant un prompt système strict (français, mise en forme Markdown, citations de sources).
5. **La réponse est renvoyée en streaming SSE** au frontend, qui l'affiche progressivement, avec :
   - le texte de la réponse ;
   - les métadonnées d'usage (tokens, coût estimé, temps de réponse) ;
   - les documents sources qui ont servi au raisonnement.

En production, Telko est pensé comme un **assistant centralisé** qui s'appuie sur :
- **Supabase** pour stocker les documents structurés ;
- **Qdrant** comme base vectorielle (RAG) pour la recherche sémantique ;
- **OpenRouter** comme passerelle vers les modèles LLM et embeddings.

---

## Description fonctionnelle du site

### Page principale « Assistant »

- **Recherche en langage naturel** : champ de saisie unique, réponses toujours en français.
- **Streaming de la réponse** : le texte s’affiche progressivement dès les premiers tokens.
- **Mise en forme Markdown** : titres, listes, gras pour les points importants, pour une lecture rapide.
- **Citations de sources** : mention explicite des documents utilisés (`[nom_du_fichier – page X]` + section `## Sources`).
- **Contexte conversationnel** : l’historique récent de la discussion est pris en compte pour affiner les réponses.
- **Feedback utilisateur (optionnel)** : possibilité de donner une note / avis sur une réponse, utilisée par le comparateur LLM.

### Gestion documentaire (backend + intégration SharePoint)

- **Ingestion manuelle** via l’API (`/embed`, `/documents`) :
  - upload de fichiers bureautiques (PDF, Word, PowerPoint…) ;
  - découpage en chunks, extraction de texte (OCR inclus) ;
  - indexation simultanée :
    - dans **Supabase** (métadonnées, texte brut) ;
    - dans **Qdrant** (vecteurs pour la recherche sémantique).
- **Synchronisation automatique SharePoint** :
  - planification via **APScheduler** ;
  - récupération des nouveaux documents / mises à jour via **Microsoft Graph** ;
  - ré-indexation transparente côté Telko.

### Administration (exposition technique)

- **Gestion des utilisateurs** (route `admin_user.py`) :
  - dépend de l’authentification **Azure AD** ;
  - permet de contrôler l’accès et, si besoin, de lier des métadonnées (département, rôle…).
- **Santé de l’API** :
  - endpoint `GET /health` pour vérifier rapidement l’état du backend (sondes de monitoring).

### Tableau de bord modèles LLM (« Comparateur de modèles »)

Accessible via la page React `LLMComparator` :

- **Vue « Activité et retours par modèle »** :
  - nombre de requêtes par modèle ;
  - latence moyenne globale et temps jusqu’au premier token ;
  - coût total et coût moyen par requête ;
  - note moyenne et taux de satisfaction calculés à partir des feedbacks utilisateurs ;
  - tri multi‑critères (par coût, vitesse, score global, etc.).
- **Vue « Catalogue OpenRouter »** :
  - prix par million de tokens (entrée/sortie) directement issus de l’API OpenRouter ;
  - fenêtre de contexte maximale (tokens) par modèle ;
  - indication de la catégorie **open‑source** / **API propriétaire** ;
  - estimation des **ressources matérielles locales** nécessaires (VRAM, parc GPU) pour ~30 utilisateurs simultanés.
- **Ressources de déploiement local** :
  - liens vers la documentation officielle pour dimensionner une éventuelle infra GPU on‑premise.

---

## Architecture applicative

### Vue globale (prod)

```text
Utilisateur (navigateur)
    │
    ▼  HTTP/S + SSE
Frontend React (Vite, Tailwind, shadcn-ui)
    │
    ▼  /api/chat, /api/llm/comparator, /api/documents...
Backend FastAPI (Python)
    │
    ├── Supabase       ──► table knowledge_documents (base documentaire « structurée »)
    ├── Qdrant         ──► collection telko_knowledge (vecteurs RAG)
    ├── OpenRouter     ──► LLM (chat) + embeddings (RAG, stats coût/token)
    └── Microsoft Graph (SharePoint) ──► ingestion & sync des documents
```

### Route de chat (`POST /chat`)

- **Entrée** : question utilisateur + métadonnées côté frontend (id de conversation, département, rôle, etc.).
- **Traitement** :
  - récupération de l'historique de la conversation ;
  - récupération de contexte documentaire (Supabase, et/ou Qdrant via `RAGPipeline`) ;
  - construction d'un prompt système très guidé (format Markdown, citations de sources, section `## Sources`) ;
  - appel LLM via OpenRouter (streaming token par token).
- **Sortie** : flux SSE contenant :
  - les tokens de texte de la réponse ;
  - un bloc final de métadonnées (`usage`) avec le détail des tokens et du coût estimé.

### Pipeline RAG (`backend/core/`)

```text
core/
  llm/              ← abstraction LLM provider-agnostique (OpenRouter)
  rag_pipeline.py   ← RAGPipeline : Qdrant + OpenRouter
  vector_store.py   ← QdrantStore (OpenRouterEmbeddings, QdrantClient)
  embeddings.py     ← (historique / legacy autour d’Ollama)
```

**RAGPipeline** orchestre :
- le découpage des documents (`RecursiveCharacterTextSplitter`) ;
- l'indexation des chunks dans Qdrant (`QdrantStore.add_documents`) ;
- la recherche sémantique (`QdrantStore.similarity_search`) ;
- la construction du message système avec les extraits documentaires déjà formatés pour l'utilisateur ;
- l'appel au LLM provider (OpenRouter) et le streaming des tokens au client.

En mémoire, un petit **historique de conversation** est conservé par `conversation_id` pour que les réponses soient cohérentes sur plusieurs tours.

### Vector store et embeddings (prod)

Le fichier `backend/core/vector_store.py` définit :
- `OpenRouterEmbeddings` : client minimal pour l’endpoint `/embeddings` d’OpenRouter,
  - gère le **cache de la dernière métrique d’usage** (tokens d’embedding) ;
  - renvoie des vecteurs de taille fixe (_VECTOR_SIZE = 1536, aligné sur `text-embedding-3-small`).
- `QdrantStore` : wrapper autour de Qdrant (client HTTP + `QdrantVectorStore`) qui fournit :
  - `init_collection()` : création de la collection si besoin (cosine, 1536 dimensions) ;
  - `add_documents()` : embedding + upsert des `Document` LangChain ;
  - `similarity_search()` : recherche sémantique, éventuellement filtrée ;
  - `delete_document()` : suppression de tous les points pour une source donnée ;
  - `get_last_embeddings_usage()` : exposition des métriques d’usage embeddings pour la couche LLM.

En production, Qdrant peut être :
- soit **hébergé en interne** (Docker / VM) ;
- soit un **cluster managé** (URL publique + clé API) configuré via les variables d'environnement.

### LLM Comparator (observabilité LLM côté frontend)

La page `frontend/src/pages/LLMComparator.tsx` expose un **tableau de bord** permettant de :
- visualiser, par modèle :
  - le nombre de runs ;
  - la latence moyenne totale et du premier token ;
  - le coût cumulé et le coût moyen par requête ;
  - la note moyenne et le taux de satisfaction (si les utilisateurs notent les réponses) ;
- comparer les **prix catalogue OpenRouter** (input/output par million de tokens) ;
- estimer l'**ordre de grandeur de la VRAM** nécessaire si l’on envisage un déploiement local du modèle.

Ce comparateur s’appuie sur une route backend dédiée (`/api/llm/comparator`) qui agrège :
- les **logs d’utilisation** du LLM (tokens, coût, timings) ;
- le **catalogue des modèles OpenRouter** ;
- des **correspondances internes “modèle → profil hardware”** maintenues dans le dépôt.

---

## Prérequis

| Outil | Version / Rôle |
|---|---|
| **Node.js** | 18+ (frontend, Vite) |
| **Python** | 3.11+ (backend FastAPI) |
| **Docker + Docker Compose** | pour Qdrant et/ou déploiement backend |
| **Accès OpenRouter** | clé API + configuration site/app |
| **Supabase** | base de données documentaire (`knowledge_documents`) |
| **Azure AD** | authentification des utilisateurs |
| **SharePoint / Microsoft 365** | source documentaire principale (via Microsoft Graph) |

En local, un simple **backend FastAPI + Qdrant + OpenRouter** suffit pour reproduire la majorité des comportements.

---

## Variables d'environnement

Fichier `.env` à la racine du dépôt (chargé automatiquement par `backend/config.py` via `pydantic-settings`).

### Backend — configuration principale

```env
# Supabase (base documentaire « structurée »)
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_ANON_KEY=...

# OpenRouter (LLM + embeddings via API)
OPENROUTER_API_KEY=...
OPENROUTER_SITE_URL=https://telko.your-company.com
OPENROUTER_APP_TITLE=Telko Assistant
OPENROUTER_LLM_MODEL=openai/gpt-4o-mini
OPENROUTER_EMBEDDINGS_MODEL=openai/text-embedding-3-small

# Qdrant (vector store RAG)
QDRANT_URL=https://your-qdrant-endpoint    # ou http://localhost:6333 en dev
QDRANT_COLLECTION_NAME=telko_knowledge
QDRANT_API_KEY=...

# Azure AD
AZURE_TENANT_ID=...
AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...

# SharePoint / Microsoft Graph
SHAREPOINT_SITE_ID=...
SHAREPOINT_DRIVE_ID=...

# CORS (origines autorisées pour le frontend)
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:8080
```

> Le champ `OPENAI_API_KEY` existe encore pour compatibilité avec l’ancienne architecture,
> mais la configuration actuelle s’appuie sur **OpenRouter** pour les appels LLM et embeddings.

### Optionnel — pipeline RAG local historique (Ollama)

Certains modules historiques supportent un pipeline RAG local basé sur **Ollama**. Pour un environnement full cloud,
vous pouvez ignorer cette section. Sinon :

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=mistral
OLLAMA_EMBED_MODEL=nomic-embed-text
```

---

## Démarrage en local (dev)

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

API disponible sur **http://localhost:8000**.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Application sur **http://localhost:8080**.
Les appels API sont proxifiés vers `http://localhost:8000` via `vite.config.ts`
tant que `VITE_API_URL` n'est pas défini.

### Qdrant (optionnel en local)

Si vous n’avez pas de Qdrant managé, vous pouvez démarrer une instance locale :

```bash
docker compose up qdrant -d
```

---

## Déploiement Docker (backend + Qdrant)

Le dépôt contient un `docker-compose.yml` qui permet de lancer le backend et Qdrant ensemble.

```bash
docker compose up --build -d
```

- le backend écoute sur le port **8000** ;
- Qdrant sur **6333** ;
- OpenRouter et Supabase restent des services **externes** accessibles via leurs URL publiques.

En production, vous pouvez :
- soit **réutiliser ce compose** sur une VM managée (type IaaS) ;
- soit **builder l’image backend** et la déployer sur un orchestrateur (Kubernetes, Cloud Run, etc.) en pointant
  vers un Qdrant/Supabase/OpenRouter managés.

---

## Structure du dépôt

```text
telko/
├── backend/
│   ├── api/
│   │   └── routes/
│   │       ├── chat.py          # POST /chat — streaming SSE (OpenRouter + Supabase + RAG)
│   │       ├── embed.py         # POST /embed — ingestion de documents
│   │       ├── documents.py     # Gestion des fichiers
│   │       ├── admin_user.py    # Administration utilisateurs
│   │       └── health.py        # GET /health
│   ├── auth/
│   │   └── azure_ad.py          # Validation tokens Azure AD
│   ├── core/
│   │   ├── llm/                 # Abstraction LLM (OpenRouter provider, BaseLLMProvider)
│   │   ├── rag_pipeline.py      # RAGPipeline — Qdrant + OpenRouter
│   │   ├── vector_store.py      # QdrantStore (OpenRouterEmbeddings + Qdrant)
│   │   └── llm_legacy.py        # DEPRECATED — ancienne implémentation LangChain
│   ├── ingestion/
│   │   ├── file_parser.py       # PDF, Word, PPTX, OCR (pytesseract)
│   │   ├── sharepoint.py        # Microsoft Graph API
│   │   └── sync_scheduler.py    # APScheduler
│   ├── scripts/
│   │   └── test_llm.py          # Test rapide generate() / stream()
│   ├── config.py                # Settings (pydantic-settings)
│   ├── main.py                  # Point d'entrée FastAPI
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                    # App React (Vite + shadcn-ui + Tailwind CSS)
├── docker-compose.yml           # Qdrant + backend
├── RUN.md                       # Mémo démarrage rapide
└── README.md
```

---

## Tester la couche LLM

Pour vérifier rapidement que la configuration OpenRouter + Qdrant est fonctionnelle, vous pouvez utiliser
le script de test :

```bash
cd backend
source .venv/bin/activate
python -m backend.scripts.test_llm
```

> Ce script teste un flux de génération et de RAG basique. En cas d’erreur, vérifier en priorité :
> - la validité de `OPENROUTER_API_KEY` ;
> - l’accessibilité de Qdrant (`QDRANT_URL`, `QDRANT_API_KEY`) ;
> - la présence de documents indexés dans la collection.

---

## Stack technique

| Couche | Technologie / Rôle |
|---|---|
| **Frontend** | React, TypeScript, Vite, Tailwind CSS, shadcn-ui |
| **Backend** | FastAPI, Python 3.11, uvicorn |
| **LLM principal (chat)** | OpenRouter (`openai/gpt-4o-mini` par défaut, configurable) |
| **Embeddings (RAG)** | OpenRouter (`text-embedding-3-small`, 1536 dims) |
| **Vector store RAG** | Qdrant (local ou managé) |
| **Base documentaire structurée** | Supabase (`knowledge_documents`) |
| **Auth** | Azure AD (tokens JWT vérifiés côté backend) |
| **Ingestion fichiers** | PyMuPDF, python-docx, python-pptx, pytesseract |
| **Sync documents** | Microsoft Graph API + APScheduler (SharePoint → Telko) |
| **Comparateur LLM** | Route `/api/llm/comparator` + page React `LLMComparator` |
| **Observabilité coût/perf LLM** | Agrégation des `usage` OpenRouter (tokens + coût) côté backend |
| **Conteneurisation** | Docker, Docker Compose |

Telko est conçu pour fonctionner aussi bien :
- **en dev local** (backend + frontend + Qdrant local + OpenRouter) ;
- qu’**en prod** sur une infra cloud (backend conteneurisé, Qdrant/Supabase/OpenRouter managés,
  intégration Azure AD/SharePoint existantes).
