# Préparer localement, entraîner par séances bornées

Direction tirée du document fourni le 26 septembre 2026. Aucun GPU distant n'est
réservé et aucun compte cloud n'est créé par ce plan. Les prix cités dans le
document ne sont pas repris comme devis vérifié.

## Répartition des responsabilités

1. `web/` et `research/` collectent des textes avec sources et budgets.
2. `knowledge/` conserve les révisions, sépare les pages et retire les doublons
   exacts. `forge/curation.rs` retire les passages identiques partagés avec les
   partitions réservées. L'import HF applique aussi des heuristiques de qualité
   et SimHash ; elles ne garantissent ni factualité ni absence de doublons proches.
3. `curriculum/` prépare des leçons. Les réponses inventées par Bailey ne sont
   pas considérées comme des vérités : exercices calculables, sources et revue
   indépendante doivent fournir la vérification.
4. `forge/dataset/` produit des shards U32 pré-tokenisés, un manifeste des sources,
   les SHA256 du tokenizer et des shards, les effectifs réels et la partition.
   Le chargeur vérifie leur intégrité et les IDs. Le constructeur reste limité à
   512 Mio de texte par partition. Le lecteur historique `shards` charge la
   partition en RAM et refuse plus de 134 217 728 tokens. `shards-stream` lit
   maintenant les fenêtres par pages de 64 Kio avec cache réglable, sans cette
   limite fixe. Son index et ses handles dépendent du nombre de shards.
5. `training/` mesure le débit et sauvegarde poids, moments AdamW, pas global et
   sélection. Le planning utilise ce pas ; le tirage des fenêtres est déterminé
   par la graine et le pas global, sans curseur de lecture mutable. `resume`
   reprend cet état dans un nouveau dossier. `--init-from` reste une nouvelle
   séance avec optimiseur neuf. La reprise CPU est testée contre un entraînement
   continu ; l'identité numérique entre matériels/backends n'est pas garantie.
6. `learning/` : comparer les candidates sur une validation fixe de sélection,
   puis conserver ou promouvoir une version avec possibilité de revenir en arrière.

Un benchmark consulté après chaque candidate est un jeu de validation. Le test
final doit rester distinct et ne pas servir à choisir la meilleure candidate.
Un pourcentage global de « connaissance » n'est pas défini : chaque mesure doit
nommer ses tâches, son nombre d'exemples et son critère de réussite.

## Conditions avant un essai cloud

- Corpus français substantiel et autorisé, avec qualité et séparation vérifiées.
- Tokenizer final figé. Changer ses IDs nécessite une nouvelle initialisation
  ou une migration explicite ; ne pas charger aveuglément les poids existants.
- Shards prêts et vérifiés avant le démarrage de la location.
- Reprise complète testée après interruption ; export et récupération testés.
- Débit et VRAM mesurés avec la précision et les lots réellement employés.
- Environnement Linux/CUDA reproductible, tests, commande de lancement bornée.
- Durée maximale, coûts estimés à partir de l'offre confirmée, arrêt et récupération
  des checkpoints. Une réservation payante sera une action distincte.

## Ordre de livraison

La priorité actuelle reste les bases françaises observables. Les diagnostics de
génération, cours versionné, warmup, décroissance du taux, clipping, premiers shards
et reprise complète sont disponibles. Les corpus pilotes comprennent 120 articles
Wikipédia et un import de 100 documents FineWeb2-HQ ; ils ne constituent pas encore
un pré-entraînement substantiel. Voir les [données françaises](../data/french-foundation.md)
et le [pilote Hugging Face](../data/huggingface.md).

La [préparation du tokenizer commun](tokenizer-generaliste.md) ajoute un mélange
train français/anglais/code et un audit de restitution/fragmentation sur sondes
de développement. Les candidats restent distincts du tokenizer des poids actifs.

Suite : ingestion Parquet avec projection du texte et révision figée, construction
des shards en flux et déduplication à grande échelle, corpus élargi et revu, choix définitif
du tokenizer, comparaison locale puis conditionnement cloud. BF16, accumulation,
cache KV et attention optimisée nécessitent leurs propres tests et mesures.

Le GPU loué sert à exécuter un travail déjà préparé. La collecte et la revue des
données restent en amont. Aucun volume de tokens ni progrès de compétence n'est
promis à partir de la seule puissance du GPU.
