# Entraînement accompagné et console visible

Le [rapport de la séance exécutée](../validation/seance-lecons-2026-09-22.md)
donne les résultats mesurés, y compris les erreurs qui restent à corriger.

## Rôle des sous-agents

Trois responsabilités ont été déléguées pour cette séance :

- Enseignement : préparer des leçons originales et vérifier leurs solutions.
- Interface : créer une console interactive avec aide et exemple.
- Relecture : vérifier la boucle d'apprentissage et les limites des mesures.

Le modèle de Bailey continue d'apprendre ses propres paramètres. Les sous-agents
fournissent des exemples et des contrôles ; ils ne lui transfèrent pas leurs poids.
Ils interviennent dans cette séance de développement, sans service de sous-agents
permanent intégré à Bailey.

## Corpus

3 360 petits exercices issus de 14 familles : arithmétique, comparaisons, minimum,
maximum, borne, divisibilité et intervalles. Chaque exemple tient dans le contexte
actuel de 128 octets. La préparation n'utilise aucun modèle ou corpus téléchargé.

Les partitions contiennent 2 688 exemples d'entraînement, 336 de validation
et 336 de test. Une constante reste dans une seule partition de ce nouveau corpus.
Toutes les partitions partagent les mêmes familles de solutions.

Les tests compilent les 3 360 solutions de référence écrites dans le projet et
vérifient 70 réponses calculées indépendamment. Les générations de Bailey ne sont
pas exécutées par cette séance.

## Déroulement

```powershell
.\tools\launch\start-visible.ps1 -Train
```

1. La fenêtre compile et vérifie le calcul GPU.
2. Elle évalue des exemples de référence sur la version précédente.
3. Elle recharge ses poids et entraîne 3 cycles de 800 étapes, lots de 16,
   taux d'apprentissage 0,0003 et graine 43.
4. Elle compare les générations sur les mêmes exercices de validation,
   puis mesure un échantillon du jeu de test.
5. Elle laisse la console ouverte pour les essais de l'utilisateur.

La reprise recharge seulement les poids et l'architecture. L'état d'Adam et
le compteur sont réinitialisés au début de la séance, puis conservés entre les
trois cycles. La sélection des sauvegardes utilise la perte de validation.

`runs/sessions/<identifiant>/` contient le journal du terminal, l'état de la
séance et les rapports `before.json`, `after.json`, `test.json`.
`runs/lessons-<identifiant>/` contient les nouveaux poids et les mesures.
Le terminal utilise sa propre copie de l'exécutable pour rester utilisable
pendant les compilations et tests ultérieurs du dépôt.

## Interpréter les contrôles de génération

Le score compte les réponses identiques à une solution connue, hormis les
espaces au début et à la fin. La longueur de réponse est fournie par la référence.
Ce contrôle est strict : un autre code équivalent peut être compté comme différent.
Il ne vérifie pas un projet entier et ne mesure pas l'autonomie d'un agent.

La sélection actuelle prend le premier exemple de chaque famille : 14 exercices
sur 336, avec constante 1 en validation et 10 en test. Ces constantes pouvaient
déjà apparaître dans le corpus de `bailey-v0`. Il s'agit d'un premier contrôle de
patrons connus, pas d'une preuve de généralisation à des constantes ou algorithmes
entièrement inconnus. Les prochains contrôles devront couvrir plusieurs groupes,
dont des constantes supérieures à 31, avant de conclure à une capacité plus large.

## Utiliser la console

- `/exemple` propose une petite addition au format appris (exemple d'entraînement).
- Une ligne de code suivie d'Entrée demande une continuation.
- `/temperature 0` rend le choix déterministe ; une valeur positive diversifie.
- `/aide` rappelle les commandes ; `/quitter` termine la boucle interactive.

Chaque essai génère au maximum 160 octets et peut continuer après la première
fonction. Les séquences littérales `\n` restent littérales ; `/exemple` inclut
déjà les retours à la ligne nécessaires.

Pour lancer les commandes séparément :

```powershell
cargo run -- prepare-lessons --out data/mes-lecons
cargo run -- grade-lessons --run runs/bailey-v0 --file data/mes-lecons/validation.jsonl --samples 14
cargo run -- console --run runs/bailey-v0
```
