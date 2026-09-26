# Séance accompagnée du 22 septembre 2026

## Travail des sous-agents

- Enseignement : 3 360 leçons originales, 14 familles, partition par constante.
- Console : interface Rust interactive, aide, exemple, réglage de température.
- Relecture : revue de l'entraînement, des générations et des mesures.

Le contrôleur principal a intégré le tout, lancé la fenêtre visible et vérifié
le déroulement. Les sous-agents n'ont transféré aucun poids à Bailey.

## Expérience exécutée

Modèle initial : `runs/bailey-v0`, lui-même entraîné depuis zéro.
Nouveau modèle : `runs/lessons-20260922-151526-086-b69d`.
Journaux et contrôles : `runs/sessions/20260922-151526-086-b69d`.

- 478 976 paramètres, contexte 128 octets.
- Corpus de 2 688 leçons d'entraînement, 336 de validation, 336 de test.
- 3 cycles de 800 étapes ; lots de 16 ; taux 0,0003 ; graine 43.
- Optimiseur neuf au début de la reprise, conservé entre les cycles.

| Mesure | Résultat |
| --- | --- |
| Perte initiale sur la nouvelle validation | 2,7441 |
| Meilleure perte de validation | 0,3875 à l'étape 1 400 |
| Perte à l'étape finale 2 400 | 0,4246 |
| Temps de la boucle, évaluations et sauvegardes comprises | 59,52 s |
| Références de validation retrouvées avant | 0 / 14 |
| Références de validation retrouvées après | 0 / 14 |
| Références retrouvées dans l'échantillon de test après | 1 / 14 |

La console recharge la meilleure version, à l'étape 1 400. Les dernières étapes
n'ont pas remplacé cette sauvegarde puisque la validation s'était dégradée.

## Ce qui a été constaté

Après cette séance, la génération commence plus facilement par une expression
Rust, mais ne respecte pas encore les constantes demandées. Par exemple :

```text
Demande : Ajouter 10 a x.
Reference : x + 10 }
Generation de la longueur attendue : x + 106
```

Les différences ne se réduisent donc pas à la mise en forme. La baisse de perte
n'a pas produit de gain sur le petit contrôle de validation par génération.

Ces contrôles prennent la première constante de chaque famille (1 en validation,
10 en test) et fournissent la longueur de la réponse. Ces constantes existaient
déjà dans le corpus initial. Ce résultat ne constitue pas une mesure de
généralisation à de nouveaux algorithmes. Les programmes générés ne sont pas exécutés.

## Vérifications techniques

- Formatage et Clippy sans avertissement.
- 13 tests automatisés réussis.
- Compilation des 3 360 solutions écrites dans le projet.
- 70 réponses de référence vérifiées indépendamment.
- Commandes de console `/aide`, `/temperature`, `/exemple`, `/quitter` vérifiées.
- Fenêtres Windows ouvertes et identifiées par leur titre.

## Prochain travail ciblé

Travailler la relation entre la demande et le corps de la fonction, notamment la
copie des constantes, avec une évaluation stratifiée sur plusieurs groupes.
Une éventuelle nouvelle recette doit être choisie sur la validation ; le jeu de
test ne doit pas servir à ajuster les paramètres ou à sélectionner des versions.
Ne pas simplement prolonger cette même séance dont la validation s'est dégradée.
