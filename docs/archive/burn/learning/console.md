# Dialogue, version chargée et arrêt

La console charge une version de Bailey au démarrage. Elle conserve ces poids
pendant toute la conversation, même si une autre fenêtre entraîne et retient une
nouvelle candidate. Il n'y a pas de rechargement automatique du modèle.

Après une session de recherche avec entraînement, le fichier
`runs/<session>/learning/active.json` indique la version retenue. Dès qu'il existe,
on peut ouvrir une nouvelle console avec cette version, depuis la racine du dépôt :

```powershell
$baileyVersion = Get-Content runs/research-02/learning/active.json | ConvertFrom-Json
.\target\release\bailey.exe console --run $baileyVersion.run
```

Le nom `research-02` est à remplacer par celui de la session. Une candidate rejetée
ne remplace pas la version indiquée dans ce fichier. Les anciennes versions restent
disponibles pour ouvrir une autre console et comparer les résultats.

## Ce qui répond dans la console

- Une entrée ordinaire utilise le mode `/dialogue` et le modèle chargé.
- `/brut` complète directement du texte ou du code avec les mêmes poids.
- `/recherche sujet` collecte des articles et affiche leurs sources.
- `/memoire sujet` affiche les articles déjà enregistrés.

Les recherches ne sont pas présentées comme des réponses produites par le modèle
et ne modifient pas ses poids. Le dialogue traite une question courte à la fois ;
aucun historique n'est ajouté au contexte. Les réponses peuvent rester incorrectes.

## Interrompre les évaluations d'une session

La boucle d'apprentissage consulte son fichier `STOP` et son budget pendant les
évaluations, avant chaque nouvel octet généré. Elle abandonne une réponse incomplète
et ne produit aucun rapport d'évaluation partiel utilisable pour une promotion.
Un calcul GPU déjà lancé doit toutefois finir avant que l'arrêt puisse être traité.

Cette interruption concerne les évaluations pilotées par la session de recherche.
La console indépendante se ferme avec `/quitter` ; elle ne surveille pas le fichier
`STOP` d'une autre session.
