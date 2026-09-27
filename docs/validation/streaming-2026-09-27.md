# Lecteur en flux : résultats vérifiés

Le lecteur Rust `shards-stream` est intégré à l'entraînement et à la reprise.
Il lit des pages de 64 Kio avec cache LRU borné, y compris quand une fenêtre
traverse plusieurs shards. Le mode historique RAM reste disponible.

## Vérifications automatisées

- 52 tests unitaires réussis avec compilation CUDA.
- 3 tests d'intégration ordinaires réussis, dont la comparaison exacte des
  tenseurs CPU entre RAM et flux, puis entre flux continu et arrêt/reprise.
- 1 test de volume exécuté explicitement, réussi ; il reste ignoré par défaut
  car il écrit et vérifie plus de 512 Mio.
- Total exécuté : **56 tests réussis**.
- Clippy sur tous les targets avec `-D warnings`, formatage et `git diff --check`
  réussis. Aucun fichier Rust supérieur à 300 lignes.

Les tests couvrent corruption, troncature, IDs invalides, mauvais tokenizer,
provenance modifiée, bornes des fenêtres, ordre LRU, immutabilité Windows et
absence de lecture des shards test pendant train/validation. Le test d'intégration
supprime et corrompt volontairement des shards test avant l'entraînement :
les lecteurs train/validation fonctionnent toujours.

Un import de livres occupait `target/debug/bailey-core.exe`. La bibliothèque
et ses tests unitaires ont été compilés normalement ; les quatre tests
d'intégration ont été copiés à l'identique dans un lanceur temporaire dépendant
de cette même bibliothèque, puis exécutés sans interrompre l'import.
L'exécutable vérifié est `runs/stream-validation-20260927-v1/bailey-stream.exe`.
Le lanceur et son verrou de dépendances restent dans le dossier de validation.

## Preuve de volume et de cache borné

`tests/stream_large.rs` crée une fixture artificielle avec un tampon de 64 Kio.
Elle sert seulement à vérifier le lecteur, pas à entraîner Bailey :

| Mesure | Résultat |
| --- | ---: |
| Tokens dans le shard train | 134 217 760 |
| Octets du shard | 536 871 040 |
| Limite de l'ancien lecteur RAM | 134 217 728 tokens, refus vérifié |
| Fenêtres vérifiées | 42 |
| Cache maximal observé | 1 048 576 octets, soit 1 Mio |
| Pages chargées | 44 |
| Octets relus pour les fenêtres | 2 818 176 |
| Scan initial d'intégrité | 8,077 s |

Les valeurs lues au début, au milieu, à la fin et à une frontière de page
correspondent au motif attendu. Les fichiers temporaires sont supprimés après
fermeture des handles. Journal :
`runs/stream-validation-20260927-v1/large-fixture.log`.

Le cache mesure les pages de tokens, **pas toute la RAM du processus**. Le
tokenizer, l'index, les handles, la fenêtre retournée et le cache de l'OS sont
distincts. Ce test ne constitue pas un benchmark de disque froid : les données
venaient d'être écrites et le système peut les garder en cache.

## Corpus réel et CUDA

`stream-check` a vérifié les 1 453 862 tokens train de Wikipédia, sur 15 shards,
puis lu 256 fenêtres de 1 025 tokens (contexte 1 024 plus une cible).
Cache plafonné et maximum observé : 1 Mio ; 240 défauts et 32 accès servis en
cache. Rapport : `runs/stream-validation-20260927-v1/wikipedia-reader.json`.

Le modèle de **100 092 672 paramètres** a ensuite tourné avec ce lecteur sur
CUDA, séquence 128, lot 2, taux 0,000001 et quatre fenêtres de validation :

- Reprise initiale des poids français, avec leur tokenizer inchangé.
- Arrêt demandé après 3 étapes dans `runs/core-stream-cuda-20260927-part1`.
- Reprise complète des poids, moments Adam, planning et tirages à l'étape 3.
- Fin à l'étape 6 dans `runs/core-stream-cuda-20260927-resumed`.

Les empreintes des poids et moments réenregistrés à l'étape 3 sont identiques.
Le compteur Adam atteint 6. Journaux et audit sont conservés dans
`runs/stream-validation-20260927-v1` ; chaque séance possède `data-reader.json`.
Cette vérification ne prétend pas démontrer une amélioration du français.
La version active de dialogue est conservée, sans promotion du modèle d'essai.

## Portée

Le cache est réglable, la reprise conserve le mode de lecture et le test final
reste exclu. Les fichiers sont vérifiés intégralement à l'ouverture avec mémoire
bornée ; ce scan représente du travail CPU/disque avant le calcul GPU.
Un handle reste ouvert par shard et l'index dépend du nombre de fichiers.

La création des shards et l'import massif restent des travaux séparés : le
constructeur actuel conserve sa limite de texte en RAM. La livraison réalise
la lecture en flux pour l'entraînement, pas encore toute la chaîne d'ingestion
en flux. Voir les [commandes et limites](../data/streaming.md).
