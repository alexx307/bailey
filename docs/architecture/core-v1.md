# Bailey Core v1 : une fondation de langage

Décision du 25 septembre 2026 : apprendre d'abord le français et le langage
général, puis spécialiser le modèle. Le prototype Burn et ses entraînements
restent conservés. Le nouveau moteur se trouve dans `crates/bailey-core` et
emploie Candle en Rust ; aucun poids de modèle externe n'est importé.

Ce document décrit les choix et leur audit. Les résultats d'exécution doivent
être lus dans `docs/validation/` : une architecture définie ne constitue pas
encore un modèle sachant converser.

## Architecture retenue

| Élément | Configuration `configs/core-100m.json` |
| --- | --- |
| Famille | Transformer décodeur, prédiction du prochain token |
| Vocabulaire cible | 32 000 tokens BPE, appris sur nos textes d'entraînement |
| Largeur | 768 |
| Couches | 12 |
| Attention | Causale, GQA : 12 têtes Q et 4 têtes K/V |
| Dimension d'une tête | 64 |
| Réseau par couche | SwiGLU, largeur intermédiaire 2 048 |
| Normalisation | RMSNorm avant les sous-couches et en sortie |
| Positions | RoPE, base 10 000 |
| Contexte cible | 1 024 tokens |
| Sortie | Matrice partagée avec l'embedding d'entrée |
| Biais linéaires | Aucun |
| Premier entraînement | FP32, AdamW, pour établir une référence vérifiable |

GQA partage les clés et valeurs entre groupes de têtes Query. Il réduit leurs
projections et un éventuel cache de génération ; il ne supprime pas le coût
quadratique de l'attention classique pendant l'entraînement.
Source : [article GQA](https://arxiv.org/abs/2305.13245).

Le nombre de paramètres vaut, avec `d=768`, `f=2048`, `L=12`, `V=32000`
et `k=4*64=256` :

```text
embedding partagé = V*d                         = 24 576 000
attention/couche  = 2*d*d + 2*d*k               =  1 572 864
SwiGLU/couche     = 3*d*f                       =  4 718 592
RMSNorm/couche    = 2*d                         =      1 536
RMSNorm finale    = d                           =        768
total            = V*d + L*(2*d*d+2*d*k+3*d*f+2*d)+d
                 = 100 092 672 paramètres
```

RoPE n'ajoute pas de paramètres appris. Ne pas compter deux fois l'embedding
partagé. Un tokenizer appris sur un petit corpus peut produire moins de
32 000 entrées : le moteur doit vérifier sa compatibilité et indiquer le
vocabulaire réellement obtenu. Il ne faut pas inventer des tokens pour faire
croire qu'un corpus minuscule remplit une cible de vocabulaire.

## Préserver les gradients avec Candle

Versions vérifiées au 25 septembre 2026 :
[Candle 0.11.0](https://docs.rs/candle-core/0.11.0/candle_core/)
et [tokenizers 0.23.2](https://docs.rs/tokenizers/0.23.2/tokenizers/).
Les versions sont fixées dans le manifeste pour rendre les essais comparables.

Une implémentation optimisée pour l'inférence peut donner des sorties correctes
tout en interrompant la rétropropagation. Dans Candle 0.11.0, `rope` et
`rope_i` emploient `apply_op3_no_bwd`. Pour entraîner Q et K, utiliser
`rope_slow` ou des opérations différentiables équivalentes.
Source : [implémentation RoPE](https://docs.rs/candle-nn/0.11.0/src/candle_nn/rotary_emb.rs.html).

Même précaution pour `ops::rms_norm` et `softmax_last_dim`, qui utilisent des
opérations sans rétropropagation. Employer `rms_norm_slow`,
`RmsNorm::forward_diff` ou une composition différentiable, ainsi que
`ops::softmax`. Le `sdpa` fusionné utilise lui aussi `apply_op3_no_bwd` :
l'attention d'entraînement doit conserver les opérations différentiables.
Ne pas déduire la compatibilité entraînement du seul nom `forward`.
Source : [opérations Candle 0.11.0](https://github.com/huggingface/candle/blob/0.11.0/candle-nn/src/ops.rs).

Les vérifications utiles portent sur la causalité, les dimensions GQA,
la présence de gradients finis dans chaque matrice Q/K/V et chaque couche,
la mise à jour réelle des poids, puis le rechargement d'un checkpoint.
Une baisse de perte peut cacher une partie du réseau figée ; elle ne suffit
donc ni à valider le moteur, ni à prouver une compétence linguistique.

## Ce que permet la machine

Relevé local : RTX 5070 Ti, 16 303 MiB signalés par `nvidia-smi`, pilote 616.92,
CUDA Toolkit 13.0.88 et Visual Studio 2022 Community avec outils MSVC
14.44.35207. `nvcc.exe` existe dans
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`, mais n'est
pas dans le PATH de la session inspectée. La carte est classée CUDA compute
capability 12.0 par [NVIDIA](https://developer.nvidia.com/cuda/gpus).

La présence des composants n'est pas une preuve qu'une compilation CUDA de
Candle fonctionne. Il faut préparer l'environnement Visual C++/CUDA du
processus, compiler, puis tester calcul et gradients sur cette carte.
Référence : [installation Windows CUDA 13.0](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-installation-guide-microsoft-windows/index.html).

Estimations arithmétiques FP32 pour 100 092 672 paramètres :

| Allocation | Taille théorique |
| --- | ---: |
| Poids seuls | 381,82 MiB |
| Poids + gradients + deux moments Adam | 1,49 GiB |
| Une matrice d'attention, lot 1, 12 têtes, contexte 1 024 | 48 MiB |
| Logits, lot 1, contexte 1 024, vocabulaire 32 000 | 125 MiB |
| Cache K/V éventuel, 12 couches, lot 1, 1 024 positions | 24 MiB |

Ces postes ne sont pas une estimation du pic total : plusieurs activations,
intermédiaires d'attention, gradients et buffers coexistent. L'optimiseur peut
aussi créer des temporaires. Un cache K/V concerne la génération et n'est pas
automatiquement une optimisation d'entraînement.

Le profil 100M paraît compatible avec des micro-lots modestes sur cette carte,
mais il faut mesurer le pic et le débit avant de choisir les lots ou la durée.
La première validation utilise quelques dizaines de tokens et un seul exemple ;
le contexte 1 024 vient ensuite. Passer ce petit essai ne valide pas la mémoire
à contexte complet. Aucun nombre de tokens par seconde n'est promis ici.

## Apprendre le français avant de suivre des instructions

Le pré-entraînement apprend la structure du texte par prédiction du prochain
token : phrases françaises variées, descriptions, explications, petites
histoires et connaissances vérifiées. Il doit recevoir beaucoup plus de
diversité que des exercices d'alphabet ou des variations d'un même patron.

Le corpus initial rédigé pour Bailey sert de démarrage et de test du pipeline.
Il ne fournit pas à lui seul la matière nécessaire pour exploiter 100M
paramètres. Répéter longtemps le même petit corpus favorise sa mémorisation.
Les travaux sur la relation entre taille du modèle et quantité de données
confirment la nécessité de raisonner sur les deux :
[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).

Après une base linguistique mesurée, les dialogues supervisés enseignent les
tours utilisateur/assistant et la réponse à une consigne. Les exemples doivent
être exacts, variés et évalués avec des formulations absentes de l'entraînement.
Lire une page web, la conserver en mémoire et entraîner les poids sont trois
opérations distinctes. Les connaissances rédigées par l'assistant deviennent
des exemples ; elles ne transfèrent pas ses propres poids dans Bailey.

Les textes sont séparés par document ou famille avant leur découpage. Le
tokenizer est appris sur le train. La validation sert aux choix de versions.
Le test reste hors entraînement et hors sélection. Conserver des questions
simples reformulées, des contrôles de cohérence française et des exercices
de calcul avec réponses vérifiées ; publier les réponses brutes et les échecs.

L'ancien prototype utilise des octets, des dimensions et une architecture
différents. Ses poids ne se chargent pas directement dans Core v1. La première
initialisation de cette nouvelle architecture reste donc aléatoire ; toutes
les leçons compatibles suivantes reprennent ses poids appris. Conserver aussi
la configuration, le tokenizer exact et l'état d'entraînement. Une reprise
des seuls poids avec Adam réinitialisé doit être annoncée comme telle.

## Ordre des expériences

1. Valider sur CPU une configuration miniature de la même architecture.
2. Sur GPU, vérifier un pas avec le profil 100M, un lot de 1 et un contexte
   court : gradients, poids modifiés, sauvegarde et rechargement.
3. Mesurer progressivement le contexte et le lot, jusqu'au profil 1 024.
4. Construire un corpus français substantiel, tracer sources et licences,
   dédupliquer, séparer train/validation/test puis figer le tokenizer.
5. Pré-entraîner, évaluer la langue, puis ajouter les dialogues supervisés.
6. Ajouter des outils, la recherche et des spécialisations avec des mesures
   comparables et une protection contre l'oubli des acquis précédents.

MoE, adaptateurs, MLA et BitNet restent des expériences futures. Un routeur
d'outils n'est pas automatiquement un MoE entraîné. Additionner les paramètres
de plusieurs spécialistes ne démontre pas la qualité d'un modèle dense de
même taille. Un MoE économise du calcul actif mais conserve des besoins de
stockage, de transfert et d'entraînement ; voir
[DeepSeek-V3](https://arxiv.org/abs/2412.19437).

BitNet explore des poids ternaires et un entraînement adapté. Son runtime
d'inférence n'est pas la preuve qu'un pré-entraînement 100B est accessible
sur 16 Go. Même 100 milliards de poids à 1,58 bit représentent environ
19,75 Go décimaux avant toute autre allocation. Tester ces pistes après une
référence dense évite de confondre gain mesuré et promesse d'architecture.
Sources : [rapport BitNet](https://arxiv.org/abs/2504.12285) et
[runtime officiel](https://github.com/microsoft/BitNet).
