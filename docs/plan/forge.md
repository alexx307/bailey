# Préparer localement, entraîner par séances bornées

Direction tirée du document fourni le 26 septembre 2026. Aucun GPU distant n'est
réservé et aucun compte cloud n'est créé par ce plan. Les prix cités dans le
document ne sont pas repris comme devis vérifié.

## Répartition des responsabilités

1. `web/` et `research/` collectent des textes avec sources et budgets.
2. `knowledge/` conserve les révisions, sépare les pages et retire les doublons
   exacts. Ajouter une déduplication des textes proches et un contrôle de qualité.
3. `curriculum/` prépare des leçons. Les réponses inventées par Bailey ne sont
   pas considérées comme des vérités : exercices calculables, sources et revue
   indépendante doivent fournir la vérification.
4. Futur `forge/dataset/` : produire des shards U32 pré-tokenisés, un manifeste
   des sources/licences, les SHA256 du tokenizer et des shards, les effectifs
   réels et la partition. Une file prête contient des données vérifiées, pas
   seulement une liste d'URLs à visiter.
5. `training/` : charger cette file avec un débit mesuré, enregistrer les poids,
   puis l'état Adam, le scheduler, l'état du générateur et la position de lecture
   pour une reprise exacte. Actuellement seuls les poids sont repris.
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
génération, cours versionné, warmup, décroissance du taux, clipping et mesures de
débit préparent les essais. Ensuite : corpus, shards, reprise exacte, comparaison
locale, puis conditionnement cloud. BF16, accumulation, cache KV et attention
optimisée nécessitent leurs propres tests numériques et mesures.

Le GPU loué sert à exécuter un travail déjà préparé. La collecte et la revue des
données restent en amont. Aucun volume de tokens ni progrès de compétence n'est
promis à partir de la seule puissance du GPU.
