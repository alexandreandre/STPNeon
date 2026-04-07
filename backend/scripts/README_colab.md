# Utiliser le GPU Google Colab comme serveur Ollama

Lance Ollama sur un GPU T4 gratuit et branche le backend STPNeon dessus via ngrok.
Utile quand tu n'as pas de GPU local ou que tu veux décharger les inférences du Mac.

---

## Étapes

### 1. Ouvrir le notebook sur Colab

Va sur [colab.google.com](https://colab.google.com) → **Fichier → Importer un notebook**
→ sélectionne `backend/scripts/colab_ollama.ipynb`.

### 2. Activer le GPU

**Runtime → Changer le type d'exécution → Accélérateur matériel : GPU T4 → Enregistrer**

### 3. Obtenir un token ngrok gratuit

Crée un compte sur [ngrok.com](https://ngrok.com) → **Your Authtoken** dans le dashboard.
Colle ce token dans la cellule 4 à la place de `TON_TOKEN_NGROK_ICI`.

### 4. Exécuter toutes les cellules dans l'ordre

**Runtime → Tout exécuter** (ou `Ctrl+F9`).

Les cellules 1 à 3 prennent quelques minutes (téléchargement des modèles).
La cellule 4 affiche quelque chose comme :

```
URL publique Ollama : https://xxxx.ngrok-free.app
Ajoute dans ton .env : OLLAMA_BASE_URL=https://xxxx.ngrok-free.app
```

### 5. Copier l'URL ngrok dans ton `.env`

Dans `.env` à la racine du dépôt :

```env
OLLAMA_BASE_URL=https://xxxx.ngrok-free.app
```

### 6. Relancer le backend

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload
```

Le backend utilise maintenant le GPU Colab au lieu du Mac.

### 7. Vérifier

La cellule 5 doit afficher une réponse de Mistral.
La cellule 6 maintient la connexion active — **garde l'onglet Colab ouvert**.

---

## Limites

| Contrainte | Détail |
|---|---|
| Durée de session Colab gratuit | ~12h max, puis l'URL ngrok change |
| Inactivité | La cellule 6 keepalive évite la déconnexion automatique |
| URL ngrok | Différente à chaque session — mettre à jour `.env` à chaque relance |
| GPU gratuit | T4 (16 Go VRAM) — suffisant pour mistral:7b et nomic-embed-text |

---

## Changer de modèle

Pour utiliser un autre modèle (ex: `llama3:8b`), modifie la cellule 3 :

```bash
!ollama pull llama3:8b
```

Et mets à jour `.env` :

```env
OLLAMA_LLM_MODEL=llama3:8b
```
