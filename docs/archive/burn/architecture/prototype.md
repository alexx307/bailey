# Architecture du premier prototype

## Modèle

Transformer décodeur : embeddings d'octets et de positions appris depuis
zéro, blocs pré-normalisés avec attention multi-tête causale et réseau
interne GELU, normalisation finale et projection vers 256 logits.

Configuration initiale : contexte de 128 octets, largeur 128, 4 têtes,
2 blocs, largeur interne 512. Elle est volontairement petite pour valider
la chaîne avant d'augmenter le coût des expériences.

La prédiction à la position t ne voit pas les octets après t. Les cibles sont
décalées d'un octet. Le modèle apprend à compléter du texte ; aucune interface
de dialogue ou capacité de raisonnement n'est supposée.

## Entraînement

Adam avec limitation de la norme des gradients, perte d'entropie croisée,
lots de fenêtres aléatoires dans le fichier d'entraînement. Les évaluations
utilisent le backend sans différentiation et des fenêtres fixes de validation.
Le jeu de test n'est jamais ouvert dans cette boucle.

Chaque amélioration produit un dossier de poids immuable. Un petit manifeste
pointe vers la meilleure version uniquement après une sauvegarde réussie.
Les poids sont enregistrés en pleine précision. La graine est conservée ;
elle ne garantit pas des résultats identiques entre GPU et CPU.

Les cycles automatiques continuent le même apprentissage. Ils n'inventent
aucun nouvel exemple et ne réécrivent pas le code de Bailey.

## Limites à lever ensuite

- Corpus pédagogique minuscule, patrons partagés entre les partitions.
- Contexte court ; génération naïve qui recalcule toute la fenêtre.
- Pas d'évaluation fonctionnelle des générations ni d'exécution de celles-ci.
- Pas de mémoire d'agent ni d'outils de modification des projets.
- L'option `--init-from` recharge les poids mais pas l'état d'Adam.
- Les sauvegardes conservées consomment de l'espace à chaque amélioration.
- La perte de validation est un indicateur de prédiction, pas de compétence.

Le [plan](../plan/roadmap.md) donne les critères pour franchir ces limites.
