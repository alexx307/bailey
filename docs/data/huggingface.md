# Pilote français FineWeb2-HQ

Le document fourni propose des textes français de Hugging Face pour préparer
un pré-entraînement plus substantiel. L'import Rust existe maintenant sous
`forge/huggingface/`. Il récupère du texte, pas un modèle préentraîné.

## Source et périmètre

- Jeu : [epfml/FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ),
  configuration `fra_Latn`, partition distante `train`.
- Accès public par [l'API rows](https://huggingface.co/docs/dataset-viewer/rows),
  avec offset et pages de 25 lignes par défaut, au plus 100 par requête.
- La fiche annonce ODC-By 1.0 pour le jeu, avec les conditions Common Crawl.
  Conserver les URLs originales : cette licence ne remplace pas les droits des
  contenus sources. Le manifeste conserve les liens de licence et de conditions.
- Contrairement à une affirmation du texte fourni, ce jeu HQ comporte lui-même
  une colonne `embeddings`. L'API peut la transférer, mais notre désérialisation
  l'ignore : aucun vecteur externe n'est conservé ou utilisé dans Bailey.
- L'API ne fige pas de révision dans notre pilote. Forge fige l'export local par
  empreintes. Un import massif devra épingler sa source et lire les colonnes
  utiles des fichiers Parquet, avec traitement en flux.

## Commandes

Utiliser un nouveau dossier à chaque exécution :

```powershell
& target/debug/bailey-core.exe hf-import `
  --out data/fineweb2-hq-fr-NOUVEAU --rows 100 --page-size 25 `
  --max-text-mb 20 --max-download-mb 128 --max-minutes 5
& target/debug/bailey-core.exe forge-build `
  --data data/fineweb2-hq-fr-NOUVEAU `
  --tokenizer runs/core-dialogue-sft-20260926-v1/model/tokenizer.json `
  --out data/forge-fineweb2-hq-fr-NOUVEAU --shard-tokens 100000
& target/debug/bailey-core.exe forge-info --data data/forge-fineweb2-hq-fr-NOUVEAU
```

L'import autorise 1 à 5 000 lignes, 1 à 200 Mio de texte, 1 à 1 024 Mio de
corps HTTP et 1 à 60 minutes. Les plafonds en Mio portent sur les corps reçus
et textes, pas les en-têtes/protocoles ni l'ensemble du dossier. Une réponse
est plafonnée à 16 Mio ; timeout réseau de 40 secondes. `STOP` et le délai sont
vérifiés entre requêtes. Une erreur HTTP arrête l'import, sans contourner le
quota et sans substituer une autre source.

Un export incomplet écrit `incomplete-manifest.json`, jamais `manifest.json` ;
Forge ne peut donc pas l'utiliser directement. Il reste conservé pour diagnostic.

## Contrôles et séparation

Les documents passent des contrôles de longueur, langue et score linguistique,
caractères, balises et répétitions. Les marqueurs de dialogue réservés sont
rejetés. Les doublons d'URL, de texte et certains voisins SimHash sont écartés.
Les passages significatifs identiques entre partitions sont rejetés.

La partition locale dépend de SHA256 de l'hôte de l'URL, avant tokenisation :
environ 80 % entraînement, 10 % validation, 10 % test. Un même hôte reste dans sa
partition ; des sous-domaines différents peuvent être séparés. Ce découpage
local ne doit pas être confondu avec la partition distante `train`.

Ce sont des heuristiques, pas une relecture indépendante, un contrôle des faits
ou une garantie de séparation sémantique. Le petit échantillon séquentiel n'est
pas représentatif du français ni des domaines. Ne pas fusionner des pilotes
sans nouvelle déduplication globale et contrôle de leurs partitions.

## Résultats du 26 septembre 2026

Le premier essai de 500 lignes s'est arrêté sur HTTP 429 après 230 documents.
`data/fineweb2-hq-fr-pilot-v1` conserve son manifeste incomplet ; il n'est pas
utilisé pour entraîner. Après réduction du nombre de requêtes, un nouvel import
de 100 lignes avec pages de 25 a abouti :

- Texte : 321 009 octets ; corps HTTP reçus : 3 298 078 octets.
- 100 documents retenus, aucun rejet dans ce petit échantillon.
- Export : `data/fineweb2-hq-fr-pilot-v2`.
- Shards : `data/forge-fineweb2-hq-fr-pilot-v2`.

| Partition locale | Documents | Tokens, tokenizer actuel | Shards U32 |
| --- | ---: | ---: | ---: |
| Entraînement | 84 | 130 688 | 2 |
| Validation | 8 | 7 517 | 1 |
| Test réservé | 8 | 7 143 | 1 |

Le vocabulaire actuel contient **1 131 entrées**. Les **32 000** de la
configuration désignent la capacité de la tête du modèle, pas le nombre appris
par ce tokenizer. Les nombres du tableau comptent les occurrences de tokens
dans les textes, pas des mots connus ou des connaissances acquises.

Le jeu complet et les 10–15 Go envisagés dans le document ne sont pas téléchargés.
Un milliard de tokens U32 occupe 4 milliards d'octets sur disque ; le corpus
peut rester sur disque à condition d'avoir un lecteur de lots en flux. Le
chargeur actuel est encore limité à une partition en RAM. Il faut terminer
cette étape avant un corpus massif ou une longue séance de pré-entraînement.
