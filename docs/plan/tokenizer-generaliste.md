# Un tokenizer commun pour Bailey

Plan du 27 septembre 2026, après lecture du document fourni par le propriétaire.
La priorité reste une fondation de langage française ; préparer le vocabulaire
au code ne remplace pas cette étape par un entraînement de code.

## Ce que signifie « universel »

Le Byte-Level BPE de Bailey conserve les 256 valeurs d'octet dans son alphabet.
Il peut donc représenter du texte UTF-8, y compris des écritures absentes du
corpus d'apprentissage. Le découpage peut cependant être très long. Cette
couverture ne donne aucune compréhension du français, du japonais ou du code :
les poids du modèle doivent apprendre ces compétences. Le rôle de ByteLevel et
de son alphabet est décrit dans la
[documentation Tokenizers](https://huggingface.co/docs/tokenizers/v0.20.3/en/api/pre-tokenizers#tokenizers.pre_tokenizers.ByteLevel).

Un vocabulaire commun permet de mélanger prose, code, nombres et symboles dans
la même séquence. Plusieurs tokenizers interchangeables imposeraient une gestion
explicite des IDs, des embeddings et du décodage. Ce n'est pas la direction v1.
Comparer plusieurs **candidats hors ligne** reste utile pour choisir le seul
tokenizer associé à une nouvelle famille de checkpoints.

À largeur 768, une table partagée de 32 000 entrées représente 24 576 000
paramètres ; 96 000 entrées en représenteraient 73 728 000. Avec le reste de
l'architecture actuelle inchangé, passer de 32 000 à 96 000 ferait passer le
total de 100 092 672 à 149 244 672 paramètres. Ce serait un autre modèle, pas
une extension gratuite du modèle 100M.

## État à préserver

- Le tokenizer du modèle de dialogue existant contient **1 131 IDs réels**.
- La table d'embeddings et la tête partagée ont une capacité de **32 000**.
- Les tokens stockés dans le corpus sont encore une autre quantité, mesurée
  après tokenisation. Une taille de vocabulaire ne donne pas cette quantité.

Les nombres et l'empreinte de l'ancien tokenizer sont consignés dans le
[rapport Forge](../validation/forge-2026-09-26.md). Demander 32 000 entrées au
BPE est un plafond : un corpus trop petit ou répétitif peut en produire moins.

Un nouveau tokenizer appris sur un mélange change les significations des IDs,
même si sa taille finale reste inférieure à la capacité de 32 000. Il ne faut
donc ni remplacer le fichier de l'ancien modèle ni charger ses poids comme s'ils
étaient compatibles. La nouvelle famille aura sa propre première initialisation,
sauf migration explicite, implémentée et évaluée séparément. Après cette première
initialisation, les séances successives reprennent ses poids appris.
L'ancien modèle et son tokenizer restent conservés et utilisables.

## Préparer, comparer, puis figer

1. Constituer un échantillon d'apprentissage diversifié, avec provenance et droits
   documentés : français courant, encyclopédique, dialogue, anglais, code et
   texte mêlant code et prose. Garder accents, indentation, nombres, symboles et
   quelques écritures variées. Des exemples artificiels courts vérifient le
   mécanisme, mais ne rendent pas le corpus représentatif à eux seuls.
2. Dédupliquer avant le mélange et préserver les partitions réservées. Les
   textes de validation et de test ne servent pas à apprendre les fusions BPE.
   Écarter les copies d'un document plutôt que les répartir dans plusieurs jeux.
3. Utiliser **70 % français, 15 % anglais et 15 % code** comme hypothèse de départ
   à comparer, pas comme optimum démontré. Définir ces parts en octets UTF-8 de
   texte retenu, sans les confondre avec des pourcentages de documents ou de
   tokens. Publier les parts réellement obtenues et les éventuelles répétitions.
   Les symboles et textes mixtes sont inclus dans ces catégories et testés
   séparément ; ils ne constituent pas un pourcentage supplémentaire implicite.
4. Apprendre dans de nouveaux dossiers des candidats sur ce mélange et sur une
   référence plus française, avec le même budget de texte. Garder les trois
   marqueurs de dialogue, l'alphabet complet et les paramètres de prétraitement
   explicites. Conserver SHA256, taille réelle, configuration et corpus source.
5. Comparer sur un jeu de **développement** indépendant : octets par token,
   nombre de tokens pour un même texte, longueurs par document, restitution
   exacte, marqueurs et exemples difficiles. Présenter les résultats par domaine
   et par source ; un score moyen ne doit pas masquer une régression française.
   Les octets par token ne comparent pas directement les compétences entre
   langues dont les caractères occupent des nombres d'octets différents.
6. Élargir les données et refaire la comparaison avant de nommer un tokenizer
   « final ». Un pilote de quelques mégaoctets peut valider les outils et révéler
   des défauts, mais ne justifie pas encore un gel pour un corpus de milliards
   de tokens. Le test final reste hors choix des candidats.
7. Après revue des mesures sur un corpus représentatif, enregistrer une décision
   de gel : tokenizer exact et son SHA256, corpus/version, vocabulaire et
   marqueurs, normalisation et rapport de sélection. Produire ensuite les
   shards correspondants et ouvrir l'expérience de modèle compatible. Le gel
   s'applique à cette famille de checkpoints ; une version future reste possible
   comme expérience distincte.

La meilleure compression sur quelques phrases n'est pas une preuve de meilleure
conversation. La validation suivante porte sur les générations du modèle et ses
compétences, après entraînement avec le tokenizer choisi. Les pertes par token
de deux tokenizers différents ne sont pas directement comparables ; une mesure
normalisée par octet et des tâches communes peuvent compléter l'évaluation.

## Budget de données : piloter les tokens, mesurer les octets

Le repère **environ 20 tokens d'entraînement par paramètre** est issu des travaux
Chinchilla sur l'allocation d'un budget de calcul. C'est une cible indicative,
pas un minimum pour apprendre ni une garantie de compétence pour Bailey. Le
papier donne Chinchilla à **70 milliards de paramètres et 1 400 milliards de
tokens**, et non 1 300 milliards comme dans le texte joint.
[Source : Hoffmann et al., table 1](https://arxiv.org/pdf/2203.15556).

| Taille indicative | Occurrences à un ratio de 20:1 |
| --- | ---: |
| 100 M paramètres | 2 milliards |
| 300 M | 6 milliards |
| 1 milliard | 20 milliards |
| 3 milliards | 60 milliards |
| 7 milliards | 140 milliards |

Ce tableau est une multiplication de planification. Les hypothèses de corpus,
de modèle et de calcul du papier ne constituent pas une mesure de Bailey.
Il faut suivre séparément le volume de tokens du corpus dédupliqué et les
occurrences effectivement présentées à l'optimiseur : revoir dix fois un petit
corpus ne crée pas dix fois plus de données uniques. Les passages superposés
et le rééchantillonnage augmentent aussi les occurrences. Le nombre de cibles
supervisées doit être distingué du remplissage, notamment en dialogue.

Le stockage U32 sans compression coûte exactement 4 octets par token hors
métadonnées : 2 milliards occupent **8 Go décimaux**, soit environ **7,45 Gio**.
Pour le texte brut, aucune conversion fixe « 5 Go = 1 milliard de tokens » n'est
retenue. Mesurer les octets par token avec le candidat choisi sur des sources
représentatives, puis extrapoler en indiquant les hypothèses. Les 10–20 Go de
texte proposés dans le document restent une enveloppe de collecte à vérifier,
pas la preuve que 2 milliards de tokens utiles seront disponibles.

## GPU : la mémoire ne suffit pas à fixer une taille de modèle

NVIDIA annonce **16 GB GDDR7** pour la RTX 5070 Ti et **288 GB par GPU** dans
le DGX B300. Un DGX B300 complet comporte huit GPU ; sa mémoire totale n'est
pas la mémoire d'un seul GPU.
[RTX 5070 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/),
[DGX B300](https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html).

Pour l'AdamW FP32 actuel, poids + gradients + deux moments représentent environ
16 octets par paramètre, avant activations, attention, temporaires, contexte CUDA
et allocations des autres applications. Le calcul ci-dessous est une estimation
des seuls états principaux, pas une mesure de VRAM maximale :

| Paramètres arrondis | États principaux FP32, Go décimaux |
| --- | ---: |
| 100 M | 1,6 |
| 300 M | 4,8 |
| 1 milliard | 16 |
| 3 milliards | 48 |
| 7 milliards | 112 |

Bailey 100M a déjà exécuté un essai CUDA local, décrit dans le rapport Forge.
Les tailles supérieures évoquées dans la pièce jointe restent des scénarios à
profiler : ce tableau ne prouve pas qu'un entraînement complet tient en mémoire,
ni qu'il finit dans un budget acceptable. Passer en BF16 ne divise pas forcément
tous les états de l'optimiseur par deux. Un coût d'inférence limité aux poids ne
décrit pas un entraînement ; même en inférence, il faut réserver de la mémoire
aux calculs et au contexte.

Conserver 100M pour comparer les pipelines et mesurer débit, pic mémoire et
qualité. Tester ensuite une taille supérieure dans une expérience bornée si ces
mesures le justifient. Aucun modèle 1B–7B, délai cloud, coût ni résultat de langage
n'est promis ici. La préparation d'un corpus massif et d'un lecteur de shards
en flux reste décrite dans le [plan Forge](forge.md) ; aucune location n'est
déclenchée par ce plan.
