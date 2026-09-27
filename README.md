# Bailey Core

Fondation de langage généraliste en Rust, avec le français comme première priorité.
Le code viendra comme spécialisation après l'acquisition et l'évaluation des bases.

Le projet actif se trouve dans `crates/bailey-core/`. L'ancien modèle Burn et
le `src/` racine ont été retirés à la demande du propriétaire. Une archive de
leurs sources reste dans `runs/archives/` ; les expériences sauvegardées restent
distinctes du nouveau modèle.

## Base retenue

- Candle 0.11, entraînement et inférence en Rust ; CUDA en option.
- 100 092 672 paramètres : 12 couches, largeur 768, GQA 12 Q / 4 KV.
- RoPE, RMSNorm, SwiGLU (largeur 2 048), embeddings d'entrée/sortie partagés.
- Vocabulaire cible 32 000 ; contexte maximal initial 1 024 tokens.
- Tokenizer BPE appris localement sur `train.txt`, aucun poids externe téléchargé.

La configuration est dans [core-100m.json](configs/core-100m.json), les choix et
limites dans [l'architecture](docs/architecture/core-v1.md) et le
[plan actuel](docs/plan/roadmap.md). MoE, BitNet et les experts restent des
expériences ultérieures.

## Vérifier et préparer la chaîne

Depuis la racine du dépôt :

```powershell
cargo run -- info
cargo run -- prepare-seed
cargo run -- tokenizer-train
cargo test --workspace
```

`prepare-seed` produit 120 petites unités françaises originales (lettres, mots,
grammaire, situations et dialogues), réparties avant l'entraînement du tokenizer.
C'est un corpus de démonstration pour vérifier le logiciel, **pas un corpus
suffisant pour préentraîner un modèle généraliste de 100 M de paramètres**.

Un petit modèle de contrôle permet de tester les sauvegardes rapidement sur CPU :

```powershell
cargo run -- train --tiny --out runs/core-smoke --steps 100
cargo run -- generate --run runs/core-smoke --prompt "Bonjour"
```

`--tiny` désigne explicitement le modèle de test. La configuration normale est
celle de 100M ; les rapports indiquent toujours le nombre de paramètres réellement
utilisé. Une sortie après quelques pas peut être incohérente.

## CUDA sur cette machine Windows

Le script initialise MSVC et CUDA uniquement dans son processus :

```powershell
.\tools\build\core-cuda.ps1
.\tools\build\core-cuda.ps1 -CargoArguments @('run','--features','cuda','--','--device','cuda','check-model','--report','runs/core-gpu-check.json')
```

`check-model` vérifie les gradients et AdamW sur trois pas synthétiques. Pour
entraîner réellement, utiliser `train` avec ses corpus `train.txt` et
`validation.txt`, puis mesurer séparément les capacités. Le jeu de test ne sert
jamais à sélectionner les checkpoints. Le CPU est le périphérique par défaut ;
`--device cuda` exige une compilation avec `--features cuda`.

La baseline utilise FP32 et une attention différentiable classique. Cache KV,
FlashAttention d'entraînement, précision mixte et accumulation des gradients
ne sont pas encore implémentés. La limite de contexte n'est pas une promesse
de débit ou de consommation mémoire à cette longueur.

## Cours de français et diagnostics

Le cours versionné contient 170 échanges d'entraînement, 28 de validation et
28 réservés au test. Il reste un exercice limité, pas un pré-entraînement complet.

```powershell
.\tools\learn\train-dialogue.ps1 -Steps 800
```

Le script reprend les poids actifs et leur tokenizer, crée une nouvelle séance
CUDA, puis enregistre des générations sur la validation. Il laisse la version
active inchangée pour permettre d'examiner le résultat. Le taux monte progressivement
puis décroît ; les gradients sont bornés et le débit est journalisé.

`train --objective dialogue` apprend sur des échanges complets indépendants,
avec une perte limitée à la réponse et au marqueur de fin. Le remplissage des
lots est ignoré dans cette perte. `--objective next-token` reste le mode normal
pour le pré-entraînement sur du texte continu. Leurs pertes ne sont pas directement
comparables : les cibles et contextes diffèrent.

Dans la console, `/diagnostic` affiche les premières prédictions et la raison
d'arrêt. `/greedy` conserve le choix déterministe ; `/sampling` active température,
top-k, top-p et pénalité de répétition. Changer ces réglages n'apprend rien au modèle.
La console indique son périphérique et ne s'entraîne pas pendant l'attente.

La commande `generate` accepte `--temperature`, `--top-k`, `--top-p`,
`--repetition-penalty`, `--seed` et `--diagnostics chemin.json`.
`dialogue-report --run chemin --prompts validation.json --out rapport.json`
conserve les questions, références, générations et diagnostics, sans donner
un pourcentage trompeur de compréhension. Les probabilités affichées sont
renormalisées sur les IDs effectivement connus du tokenizer.

Le [plan Forge](docs/plan/forge.md) décrit la préparation locale des données
avant de futures séances sur GPU distant. La location et le lancement cloud
ne sont pas encore implémentés. Le [rapport Forge](docs/validation/forge-2026-09-26.md)
documente les corpus pilotes, les 41 tests et l'arrêt/reprise CUDA du modèle 100M.

## Reprendre les acquis

Pour ouvrir une fenêtre de console sur la version active :

```powershell
.\tools\launch\start-core.ps1
```

Les [premiers résultats](docs/validation/core-2026-09-26.md) décrivent les essais
100M et la boucle Internet. Le [cours de dialogue](docs/validation/dialogue-2026-09-26.md)
permet maintenant quelques réponses enseignées correctes, mais beaucoup de
reformulations échouent encore. Le modèle reste expérimental et n'a pas reçu un
pré-entraînement suffisant pour une conversation générale.

```powershell
cargo run -- train --init-from runs/core-smoke --tokenizer runs/core-smoke/tokenizer.json --out runs/core-suite
cargo run -- console --run runs/core-suite
```

`train --init-from` charge les meilleurs poids et l'architecture précédente ; le
tokenizer doit être identique. L'optimiseur est recréé pour cette nouvelle séance.
La commande `resume` reprend au contraire une séance interrompue, avec les poids
de la dernière étape sauvegardée, les moments AdamW, le planning et les tirages
des lots. Elle exige les nouveaux checkpoints complets et un nouveau dossier :

```powershell
& target/debug/bailey-core.exe --device cuda resume `
  --run runs/seance-interrompue --out runs/seance-reprise
```

Les checkpoints anciens sans état Adam restent utilisables avec `--init-from`.
Le format/tokenizer des anciens poids Burn ne correspond pas à ce nouveau cœur.

Créer un fichier `STOP` dans un dossier d'entraînement arrête entre deux étapes
et sauvegarde l'état complet. `--stop-after N` permet aussi un arrêt programmé
sans modifier le planning total. `latest.json` désigne la reprise et `best.json`
la meilleure validation. Chaque snapshot 100M FP32 avec les deux moments occupe
environ 1,12 Gio ; les versions précédentes sont conservées. Prévoir l'espace
disque et un intervalle `--eval-every` adapté.

## Corpus et nombre de tokens

```powershell
& target/debug/bailey-core.exe tokenizer-info --tokenizer data/core-tokenizer
& target/debug/bailey-core.exe forge-info --data data/forge-wikipedia-fr-v1
```

Le tokenizer actuel compte **1 131 entrées**, la tête du modèle dispose de
32 000 places, et le corpus Wikipédia préparé contient **1 453 862 tokens
d'entraînement**. Ces trois nombres mesurent des choses différentes.

`hf-import` importe un pilote français FineWeb2-HQ avec sources et budgets.
`curate-text` retire les passages identiques aux partitions réservées en conservant
l'original. `forge-build` transforme les textes en shards U32, puis le chargeur
vérifie leurs empreintes et leur tokenizer. Voir le [guide Hugging Face](docs/data/huggingface.md)
et le [corpus Wikipédia](docs/data/french-foundation.md).

Pour une nouvelle expérience CUDA à partir des poids français actuels :

```powershell
& target/debug/bailey-core.exe --device cuda train `
  --data data/forge-wikipedia-fr-v1 --data-format shards `
  --tokenizer runs/core-dialogue-sft-20260926-v1/model/tokenizer.json `
  --init-from runs/core-dialogue-sft-20260926-v1/model `
  --out runs/core-forge-NOUVEAU --steps 100 --stop-after 40 `
  --sequence 128 --batch-size 2 --learning-rate 0.00001 `
  --warmup-steps 10 --min-lr-ratio 0.1 --eval-every 50
```

Les shards actuels servent au texte continu (`next-token`), pas au dialogue
masqué. Le prototype charge la partition en RAM et n'est pas encore un lecteur
en flux pour des milliards de tokens. Aucun de ces pilotes ne remplace un
pré-entraînement substantiel ni une évaluation des compétences.

## Préparer le tokenizer commun

Un seul tokenizer commun est visé pour le français, l'anglais et le code.
Le [plan du tokenizer généraliste](docs/plan/tokenizer-generaliste.md) précise
la comparaison de candidats avant gel, la compatibilité des IDs et le budget
de données. Apprendre ce découpage utilise le CPU et ne modifie pas les poids
du modèle de langage.

Après les collectes françaises et anglaises documentées, le script prépare une
liste explicite de sources avec empreintes et un snapshot du code local :

```powershell
.\tools\tokenizer\prepare-inputs.ps1 -Out data/tokenizer-inputs-NOUVEAU
& target/debug/bailey-core.exe tokenizer-mix `
  --config data/tokenizer-inputs-NOUVEAU/mix.json --out data/tokenizer-mix-NOUVEAU
& target/debug/bailey-core.exe tokenizer-train `
  --train data/tokenizer-mix-NOUVEAU/train.txt `
  --out data/tokenizer-candidate-NOUVEAU --vocab-size 32000
& target/debug/bailey-core.exe tokenizer-audit `
  --tokenizer data/tokenizer-candidate-NOUVEAU `
  --baseline runs/core-dialogue-sft-20260926-v1/model/tokenizer.json `
  --out runs/tokenizer-audit-NOUVEAU.json
```

Le mélange vise 70 % français, 15 % anglais et 15 % code en octets sources,
sans répétition pour compenser un manque. Il conserve offsets, provenance et
empreintes ; ce pilote n'effectue pas de déduplication sémantique globale.
L'audit compare les mêmes 32 sondes de développement dans huit domaines,
la restitution exacte, les 256 symboles ByteLevel et les marqueurs. Il ne
promet pas une meilleure conversation. Les nouveaux IDs exigent une expérience
de modèle compatible ; le candidat ne remplace pas automatiquement le tokenizer
des checkpoints existants.

## Lecture Internet

```powershell
cargo run -- research-topic "langue française"
cargo run -- research --config configs/research.json --out runs/research-01
```

La collecte conserve les pages Wikipédia avec URL, révision, langue et licence.
Les sujets se configurent librement. `/recherche sujet` et `/memoire sujet` sont
disponibles dans la console. La lecture enrichit une bibliothèque locale.

Pour entraîner aussi des candidates, ajouter `--init-from chemin-du-modele`
et `--rehearsal chemin-des-bases`. Cette boucle expérimentale conserve des
validations figées et rejoue les bases pour limiter l'oubli. Elle mesure des
pertes de prédiction, pas la compréhension des articles. Voir le
[guide Internet](docs/learning/internet.md).

## Organisation

```text
crates/bailey-core/src/
  model/          réseau, attention, positions, normalisation
  tokenization/   apprentissage et chargement BPE
  curriculum/     corpus français initial et partitions
  forge/          import HF, curation, shards et intégrité
  training/       lots, AdamW, évaluation, checkpoints
  inference/      génération et console
  web/            API Wikipédia
  knowledge/      bibliothèque, provenance, export
  research/       collecte et budgets
  learning/       candidates et conservation des bases
  app/            commandes et diagnostic
configs/          architecture et sujets de recherche
tools/            compilation, lancement et contrôles
docs/             plan, architecture, résultats
```

Les conventions se trouvent dans [AGENTS.md](AGENTS.md).
`tools/quality/check.ps1` contrôle formatage, Clippy, tests et fichiers Rust
d'environ 300 lignes maximum.
