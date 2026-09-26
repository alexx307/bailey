# Corpus documentaire français — fondation v1

Cette collecte prépare des textes pour Bailey. Elle ne charge aucun modèle et ne
modifie aucun poids. Le corpus encyclopédique est distinct des leçons de dialogue.
Il reste beaucoup trop petit pour établir à lui seul un pré-entraînement français
généraliste de 100 millions de paramètres.

## Collecte reproductible

La configuration `configs/research-french-foundation.json` demande 40 sujets,
trois résultats au plus par sujet, un seul cycle, une seconde entre requêtes,
un plafond de 20 minutes et 100 000 000 octets de bibliothèque. Les sujets couvrent
la langue, la culture, les mathématiques et des sciences élémentaires. Les trois
résultats d'une recherche peuvent être spécialisés ou seulement liés au sujet.

```powershell
& target/debug/bailey-core.exe --device cpu research `
  --config configs/research-french-foundation.json `
  --library data/library-french-foundation-v1 `
  --out runs/forge-corpus-20260926-v1
```

Le CPU suffit à la collecte réseau. L'absence de `--init-from` exclut tout
entraînement automatique. Une nouvelle exécution doit utiliser un nouveau dossier
de session ; les anciens corpus et poids restent conservés.

## Traçabilité et séparation

- Bibliothèque nouvelle : `data/library-french-foundation-v1/`.
- Journal : `runs/forge-corpus-20260926-v1/events.jsonl`.
- Export : `runs/forge-corpus-20260926-v1/datasets/cycle-0001/`.
- Le manifeste conserve page, révision, langue, titre, URL, date de collecte,
  indication de licence et empreintes SHA256. L'indication CC BY-SA et ses liens
  proviennent du collecteur ; conserver ces attributions avec les textes.
- La partition dépend de `SHA256(langue:identifiant_page)`, avant les fenêtres de
  tokens. Une page et ses révisions restent dans la même partition : environ
  80 % entraînement, 10 % validation, 10 % test, sans rééquilibrage ultérieur.
- Les doublons de révision et de texte intégral sont exclus. Les reprises
  partielles, citations communes, traductions et textes proches ne sont pas
  détectés par ce mécanisme.
- Le texte du test ne sert ni à l'entraînement, ni au choix des candidates, ni
  à la rédaction de nouvelles leçons. Un audit des effectifs et métadonnées ne
  constitue pas une évaluation du modèle.

## Résultats de la session

Collecte terminée : **120 articles**, dont 101 en entraînement, 9 en validation
et 10 en test. L'export brut contient respectivement 3 054 498, 199 829 et
363 251 octets de texte.

Le contrôle Forge a d'abord rejeté des passages identiques entre partitions.
`curate-text` a créé `data/wikipedia-fr-curated-v1` : 22 fragments retirés de
l'entraînement et 2 de la validation. Le test est conservé octet pour octet.
Les SHA256 des fragments retirés et des fichiers avant/après sont dans le nouveau
manifeste, avec la provenance originale. Aucun exemple du test n'a servi à
entraîner ou évaluer le modèle.

Les shards `data/forge-wikipedia-fr-v1` utilisent le tokenizer appris de
**1 131 entrées**, sans changement d'IDs :

| Partition | Octets texte après curation | Tokens | Shards U32 |
| --- | ---: | ---: | ---: |
| Entraînement | 3 052 696 | 1 453 862 | 15 |
| Validation | 199 673 | 89 874 | 1 |
| Test réservé | 363 251 | 164 246 | 2 |

```powershell
& target/debug/bailey-core.exe curate-text `
  --data runs/forge-corpus-20260926-v1/datasets/cycle-0001 `
  --out data/wikipedia-fr-curated-NOUVEAU
& target/debug/bailey-core.exe forge-build `
  --data data/wikipedia-fr-curated-NOUVEAU `
  --tokenizer runs/core-dialogue-sft-20260926-v1/model/tokenizer.json `
  --out data/forge-wikipedia-fr-NOUVEAU --shard-tokens 100000
```

## Portée et suite

L'API fournit du texte d'articles, pas une garantie d'exactitude, d'actualité,
de simplicité pédagogique ou d'équilibre des domaines. La langue `fr` identifie
le wiki de provenance ; aucun classifieur linguistique indépendant n'est appliqué.
Les résultats ne sont donc pas présentés comme un corpus entièrement relu.

Avant de grossir la collecte ou d'en tirer un long entraînement : ajouter une
déduplication des textes proches, examiner un échantillon de l'entraînement et
mesurer longueurs et couverture du tokenizer. Les shards et leurs empreintes sont
disponibles ; ils ne rendent pas le corpus pédagogiquement vérifié. Le tokenizer
actuel doit rester figé pour reprendre les poids existants. Un nouveau tokenizer
exige une expérience distincte ou une migration explicitement testée.
