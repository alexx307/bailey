# Plan de Bailey — fondation générale, puis spécialités

Direction confirmée le 26 septembre 2026 : construire notre modèle de langage
en Rust, apprendre d'abord le français, puis ajouter les compétences de code.

## 1. Fondation technique

Cœur Candle 100M : GQA, RoPE, RMSNorm, SwiGLU, embeddings partagés. Tokenizer BPE
local, corpus séparés, entraînement AdamW, sauvegarde SafeTensors et reprise des
poids. Le modèle réduit sert uniquement aux tests rapides ; il ne remplace pas
la cible 100M. Vérifier causalité, gradients de chaque matrice, absence d'écrasement,
sauvegarde/rechargement et consommation réelle avant d'allonger les séances.

État : implémenté ; suivre les résultats dans docs/validation. L'ancien cœur Burn
a été retiré du code actif, ses sources archivées dans runs/archives.

## 2. Pré-entraînement français

Assembler un corpus substantiel de textes français autorisés : descriptions,
récits, explications, dialogues naturels, vocabulaire et grammaire. Conserver
origine, licence, date et partition ; retirer doublons et textes dégradés.
Réserver validation/test avant apprentissage du tokenizer. Les 120 unités
actuelles ne servent qu'à vérifier la chaîne.

Former le tokenizer définitif sur ce corpus d'entraînement ; figer ses IDs.
Comparer d'abord plusieurs essais courts, mesurer tokens/seconde et VRAM,
puis calculer le budget réaliste d'une séance longue. Ne pas annoncer une durée
d'apprentissage ou une maîtrise générale sans ces mesures.

Critère : prédiction sur textes réservés et générations françaises cohérentes
évaluées sur des questions indépendantes, pas seulement une perte en baisse.

## 3. Apprentissage du dialogue

Paires question/réponse rédigées et relues, identité, salutations, explications,
reformulations et gestion honnête des informations manquantes. Apprendre les
marqueurs des rôles et la fin de réponse. Puis exemples à plusieurs tours et
gestion explicite de la longueur de contexte.

Critère : réponses utiles à des formulations nouvelles, fin de réponse correcte,
mesure séparée des erreurs factuelles, des calculs et du français.

Un cours original versionné dans `assets/curricula/french-dialogue-v1/` ajoute
170 échanges train et deux partitions de 28 reformulations. Il sert à un essai
limité de dialogue et ne remplace pas la phase de pré-entraînement substantiel.
Sampling et diagnostics top-k sont séparés des modifications de poids.
Un objectif de dialogue supervise les réponses sur des échanges isolés ; la
validation utilise des échanges indépendants avec le même masquage.
Les [mesures du cours](../validation/dialogue-2026-09-26.md) montrent des réponses
enseignées correctes et encore de nombreuses erreurs sur les reformulations.
Cette étape reste partielle ; le critère de dialogue général n'est pas atteint.

## 4. Mémoire et outils

La bibliothèque Wikipédia et la collecte bornée existent. La consultation conserve
des sources ; elle ne modifie pas les poids. Relier ensuite des passages pertinents
au modèle lorsqu'il sait les exploiter. Ajouter calculatrice et autres outils
avec des résultats identifiés comme provenant de l'outil.

Une boucle expérimentale peut déjà entraîner des candidates en rejouant les bases
et en comparant des pertes sur les mêmes validations figées. Cette sélection
statistique doit être complétée par des tests de compétences avant de devenir un
mécanisme de mise à jour de confiance. STOP, journaux et budgets restent obligatoires.

## 5. Code et vérification

Introduire ensuite exercices de programmation, compilateur, tests et réparation
dans un espace isolé. Mesurer des tâches réservées indépendantes. Le modèle ne
doit pas changer les tests qui déterminent sa propre réussite.

## 6. Optimisations et évolution

Mesurer séparément précision mixte, accumulation de gradients, attention optimisée
avec backward, cache KV, puis adaptateurs et experts. MoE et BitNet restent des
expériences : moins de poids actifs ou une inférence quantifiée ne suppriment pas
le coût des données, gradients, états d'optimiseur ou transferts mémoire.

L'auto-modification du logiciel appartient à un superviseur externe, avec
candidates distinctes, tests, promotion et retour à la version précédente.

La préparation locale des futures séances GPU distantes est détaillée dans
[le plan Forge](forge.md). Shards, reprise complète d'Adam et lancement cloud
restent à implémenter ; aucune location n'est lancée.
