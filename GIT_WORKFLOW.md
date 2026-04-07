# Git Workflow — Projet STPNeon

## Branches

| Branche | Rôle |
|---------|------|
| `main` | Code stable, ne jamais travailler directement dessus |
| `dev-mathieu` | Branche de développement principale |

## Commandes essentielles

### Démarrer une session de travail

```bash
git checkout dev-mathieu      # Se placer sur la bonne branche
git pull                      # Récupérer les derniers changements
npm run dev                   # Lancer le frontend (port 8080)
```

### Vérifier où on en est

```bash
git branch                    # Voir la branche active (marquée *)
git status                    # Voir les fichiers modifiés / non suivis
```

### Sauvegarder son travail

```bash
git add .                     # Ajouter tous les changements
git commit -m "description"   # Créer un commit
git push                      # Envoyer sur GitHub
```

### Créer une nouvelle branche de feature

```bash
# ✅ Bonne méthode : créer ET basculer en une commande
git checkout -b ma-feature

# ❌ Éviter : crée la branche sans basculer dessus, puis erreur au checkout
git branch ma-feature         # crée
git checkout -b ma-feature    # → fatal: already exists
```

## Erreurs courantes

### `fatal: a branch named 'X' already exists`
Tu as fait `git branch X` puis `git checkout -b X`. Utilise à la place :
```bash
git checkout X        # basculer sur une branche existante
# ou
git checkout -b X     # créer ET basculer (seulement si elle n'existe pas encore)
```

### `zsh: command not found: gitbranch`
Les commandes git s'écrivent avec un espace : `git branch`, pas `gitbranch`.

## Lancer le projet

```bash
# 1. Supprimer lovable-tagger si présent (fait une seule fois)
npm uninstall lovable-tagger

# 2. Installer les dépendances
npm install

# 3. Lancer en dev
npm run dev
# → http://localhost:8080
```

## Avertissements npm à ignorer

Ces warnings sont **normaux** et n'empêchent pas le projet de tourner :

- `deprecated whatwg-encoding` — dépendance indirecte de pdfjs
- `deprecated domexception` — idem
- `deprecated abab` — idem
- `Browserslist data is X months old` → fix optionnel : `npx update-browserslist-db@latest`
- `15 vulnerabilities` → fix optionnel : `npm audit fix` (ne pas utiliser `--force` sans vérifier)

## Pousser une branche sur GitHub

```bash
# Première fois (crée le tracking)
git push -u origin dev-mathieu

# Les fois suivantes (le tracking est en place)
git push
```

## Créer une Pull Request

Après un `git push`, GitHub affiche un lien direct :
```
remote: Create a pull request for 'dev-mathieu' on GitHub by visiting:
remote:      https://github.com/alexandreandre/stpneon/pull/new/dev-mathieu
```
Clique dessus pour ouvrir la PR vers `main`.
