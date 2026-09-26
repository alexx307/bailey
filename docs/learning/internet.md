# Recherche et apprentissage expérimental

Le fournisseur actuel est Wikipédia en HTTPS, via son API publique.
Les sujets se configurent librement ; aucune liste de thèmes n'est imposée.
Les textes restent des données, jamais des commandes exécutées.

## Lire et conserver

```powershell
cargo run -- research-topic "langue française"
cargo run -- research --config configs/research.json --out runs/research-01
```

La deuxième commande collecte par cycles et conserve URL, révision, licence,
langue et date dans data/library. Elle limite requêtes, délais, réponses,
stockage et durée. Un fichier STOP dans le dossier de session interrompt les
pauses immédiatement ; une requête réseau déjà en cours peut attendre son timeout
de 20 secondes. Le journal events.jsonl décrit les étapes et les erreurs.

Dans la console : /recherche sujet consulte Internet, /memoire sujet consulte
les pages conservées. Les extraits et les réponses du modèle sont distingués.

## Entraîner une candidate

```powershell
cargo run -- research --config configs/research-demo.json --out runs/research-learn-01 --init-from runs/core-smoke-20260926 --rehearsal data/core-seed --steps 20
```

Cet exemple utilise le petit modèle de contrôle pour vérifier la boucle. Pour
une véritable séance, choisir une version déjà entraînée et ses bases vérifiées.
CUDA exige --features cuda et --device cuda, comme pour train.

Les pages sont assignées par identifiant à train/validation/test ; toutes leurs
révisions restent dans la même partition. L'export attend de disposer de pages
dans chaque partition. Les doublons textuels exacts sont écartés, mais pas tous
les doublons proches ni les traductions.

La première validation web et celle des bases sont figées pour toute la session.
Chaque candidate reprend les poids actifs avec le même tokenizer et un optimiseur
neuf. Le corpus rejoue au moins autant d'octets des bases que de nouveaux textes.
Une promotion exige une amélioration web et une perte sur les bases inférieure
à 102 % de celle de la version active **et** de celle du début de session.
Le test reste exclu de l'entraînement et de cette décision.

Cette décision mesure seulement la prédiction de tokens sur quelques fenêtres ;
elle ne prouve pas la compréhension, la véracité ou la conservation de toutes les
compétences. Les candidates restent séparées ; learning/active.json désigne celle
retenue. Une console déjà ouverte garde ses propres poids jusqu'à sa fermeture.

STOP et le budget sont vérifiés entre étapes et évaluations. Un calcul GPU
déjà lancé doit se terminer. Les poids acquis restent sauvegardés ; une candidate
interrompue n'est pas promue.

## Sources

API : https://www.mediawiki.org/wiki/API:Search et
https://www.mediawiki.org/wiki/Extension:TextExtracts .
Les métadonnées de licence et d'attribution accompagnent les exports.
La collecte n'est pas un accès illimité à tous les sites, et le modèle ne sait
pas encore identifier fiablement ses propres lacunes pour choisir seul les sujets.
