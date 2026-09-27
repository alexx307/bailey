# Lecture des shards en flux

`--data-format shards-stream` lit les tokens depuis les fichiers U32 de Forge,
sans matérialiser toute la partition ni un shard entier dans un `Vec<u32>`.
Ce mode sert à l'entraînement `next-token` et à sa validation. Le dialogue
masqué conserve son chargeur d'échanges complets.
Les [résultats vérifiés](../validation/streaming-2026-09-27.md) comprennent un
shard de plus de 512 Mio, les comparaisons CPU et une reprise CUDA du modèle 100M.

## Fonctionnement

1. Lire et vérifier le manifeste, le tokenizer et la provenance.
2. Parcourir uniquement les shards de la partition demandée avec un tampon
   de 64 Kio : SHA256, nombre d'octets, IDs présents dans le tokenizer et capacité
   du modèle. Ce scan initial prend du temps disque/CPU, sans charger la partition.
3. Conserver l'index des plages de tokens et les fichiers ouverts.
4. Pour chaque lot, tirer les mêmes positions qu'avec le lecteur RAM, puis lire
   les pages de 64 Kio nécessaires. Une fenêtre peut traverser pages et shards.
5. Évincer les pages les moins récemment utilisées avant de charger une page
   qui dépasserait le budget. La mémoire des pages reste bornée.

Le cache par partition vaut 8 Mio par défaut ; `--shard-cache-mib` accepte 1 à
256 Mio dans la commande d'entraînement. L'API Rust accepte aussi 64 Kio.
Train et validation ont chacun leur cache. Le tampon de vérification, les lots,
le tokenizer, l'index, les handles et les allocations du modèle sont séparés.
Le système d'exploitation peut conserver ses propres pages en mémoire.

La limite historique de 134 217 728 tokens ne s'applique pas au lecteur en flux.
Les offsets disque utilisent 64 bits ; les comptes exposés utilisent `usize`,
donc utiliser une cible 64 bits pour les grandes partitions. L'index et le nombre
de fichiers ouverts augmentent avec le nombre de shards, pas avec leurs octets.
Un grand nombre de shards peut atteindre la limite de handles de l'OS.

## Commandes

La version vérifiée le 27 septembre est aussi disponible immédiatement dans
`runs/stream-validation-20260927-v1/bailey-stream.exe`. Un import de livres
occupait l'exécutable principal pendant la compilation ; cette entrée séparée
utilise la même bibliothèque Rust et permet de poursuivre sans arrêter l'import.
Dans les commandes ci-dessous, ce chemin peut remplacer `target/debug/bailey-core.exe`.

Vérifier les accès et le cache, sans ouvrir de modèle :

```powershell
& target/debug/bailey-core.exe stream-check `
  --data data/forge-wikipedia-fr-v1 --cache-mib 1 `
  --sequence 1024 --windows 256 --out runs/stream-check-NOUVEAU.json
```

Commencer un essai borné avec les poids et le tokenizer compatibles actuels :

```powershell
& target/debug/bailey-core.exe --device cuda train `
  --data data/forge-wikipedia-fr-v1 --data-format shards-stream --shard-cache-mib 1 `
  --tokenizer runs/core-dialogue-sft-20260926-v1/model/tokenizer.json `
  --init-from runs/core-dialogue-sft-20260926-v1/model `
  --out runs/stream-train-NOUVEAU --steps 6 --stop-after 3 `
  --sequence 128 --batch-size 2 --learning-rate 0.00001 --eval-every 6
& target/debug/bailey-core.exe --device cuda resume `
  --run runs/stream-train-NOUVEAU --out runs/stream-resume-NOUVEAU
```

`resume` conserve le mode de lecture et le planning de sa séance. Le contenu du
cache n'a pas à être sauvegardé : il n'influence ni les tokens ni les positions
tirées, déterminées par graine et pas global. La reprise revalide les fichiers.
Elle refuse une modification des données ou des réglages sauvegardés.

Le mode historique `--data-format shards` reste disponible pour les anciens
essais et comparaisons ; il conserve son chargement et sa limite RAM. La nouvelle
option de cache par défaut ne change pas la sérialisation des anciennes
configurations, afin de préserver leurs signatures de reprise.

## Intégrité et mesures

Les fichiers test ne sont pas ouverts lors du chargement de train ou validation.
Les métadonnées des trois partitions restent vérifiées dans le manifeste.
Le lecteur calcule aussi le SHA256 de tous les octets de la partition pour
détecter train et validation identiques sans comparer deux gros vecteurs.

Sur Windows, les handles des shards restent ouverts avec partage de lecture
seul : écriture, suppression et remplacement sont refusés pendant leur usage.
Ailleurs, taille et date de modification sont contrôlées à chaque accès, y
compris pour une page en cache. Il faut maintenir les fichiers immuables ; ce
contrôle ne protège pas contre une altération délibérée restaurant la même date.

`data-reader.json` décrit les lecteurs d'une séance. Les rapports distinguent :

- `validated_bytes` : octets parcourus par le scan d'intégrité initial ;
- `bytes_read` : octets lus pour charger des pages après ce scan ;
- `cache_hits` / `cache_misses` : accès aux pages ;
- `cached_bytes` / `peak_cached_bytes` : pages courantes et maximum observé ;
- `cache_limit_bytes` et `open_shards` : budget et fichiers ouverts.

Ces compteurs ne mesurent pas toute la RAM du processus, la VRAM, ni un débit
d'entraînement généralisable à d'autres matériels. `stream-check` n'évalue
aucune capacité linguistique et ne change aucun poids.

## Périmètre restant

Cette livraison concerne la lecture des shards pendant l'entraînement. Le
constructeur `forge-build` traite encore un fichier texte par partition en RAM,
avec sa limite de 512 Mio ; l'import massif Parquet et une construction des
shards entièrement en flux restent distincts. Aucun téléchargement de plusieurs
gigaoctets ni réservation de GPU distant n'est déclenché par ce lecteur.
