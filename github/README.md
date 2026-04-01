## GRAND I : Travailler sur une branche feature

# 0) Pour se mettre à jour avec main 
Mathieu : 
git stash
git checkout main
git pull origin main
git checkout dev-mathieu
git rebase main

Alex : 
git stash
git checkout main
git pull origin main
git checkout dev-alex
git rebase main


# 0) Voir ce que la feature apporte à main (à faire AVANT merge, idéalement)

git status


# 1) Créer une nouvelle branche à partir de la branche courante (idéalement main à jour)

git branch dev-prénom

# 2) Vérifier sur quelle branche on se trouve

git branch

# Si tu n’es pas dessus, basculer :

git checkout dev-prénom


# 3) Vérifier l’état du répertoire de travail (ce qui est modifié / non suivi)

git status

# 4) Ajouter au “staging” tout ce que tu veux inclure dans le commit

git add .




# 5) Créer le commit 

git commit -m "description courte du changement"



# 6) Pousser la branche sur GitHub (remote) pour la sauvegarder / ouvrir une PR

git push -u origin dev-prénom





## GRAND II : Merge avec main


# 7) Revenir sur main pour intégrer la feature

git checkout main


# 8) Mettre main à jour depuis le remote (important avant de merge)

git pull origin main


# 9) Fusionner la branche de feature dans main

git merge dev-prénom


# 10) Pousser main sur GitHub (le merge devient effectif sur le remote)

git push origin main


# 11) Vérification finale : il ne doit rien rester en attente

git status


