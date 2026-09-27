# Sources et mélange du tokenizer commun

Le mélange forme un candidat Byte-Level BPE. Il ne sert pas à apprendre les
poids de Bailey et ne change pas le tokenizer des modèles sauvegardés.

## Sources explicites

Le script `tools/tokenizer/prepare-inputs.ps1` prépare un inventaire, puis la
commande Rust `tokenizer-mix` vérifie et assemble les sources :

- `data/wikipedia-fr-curated-v1/train.txt`, avec son manifeste d'attribution.
- `data/wikipedia-en-tokenizer-curated-v1/train.txt`, avec sa provenance.
- Les 11 textes originaux listés dans `assets/tokenizer/corpus-v1/manifest.json` :
  anglais thématique, Rust, Python et C. Total : 31 683 octets, dont 14 106 anglais
  et 17 577 code. Les programmes sont des exemples vérifiés, pas un corpus massif.
- Un snapshot de 35 fichiers d'implémentation de Bailey : dossiers explicitement
  nommés, hors fichiers `*test*` et hors fichiers contenant `#[cfg(test)]`.
  Le code du tokenizer, les sondes et les cours de dialogue ne sont pas inclus.
  Ce snapshot local n'est pas téléversé et ne prétend pas attribuer une licence
  publique aux sources privées du projet.

Les fichiers texte français et anglais proviennent uniquement des partitions
train. La déclaration train des sources dans la configuration est une frontière
de confiance : ne pas y déclarer arbitrairement un jeu réservé. Chaque fichier
doit correspondre à son SHA256 attendu. Une provenance signalée incomplète est
refusée. Les références à `test.txt` et `validation.txt` sont rejetées.

## Collecte anglaise du 27 septembre 2026

`configs/research-english-tokenizer.json` demande 10 sujets, 3 articles par sujet,
un cycle, une seconde de délai, 10 minutes et 20 000 000 octets de bibliothèque.
La collecte achevée contient 30 articles : 23 train, 5 validation, 2 test.

```powershell
& target/debug/bailey-core.exe research `
  --config configs/research-english-tokenizer.json `
  --library data/library-english-tokenizer-NOUVEAU `
  --out runs/tokenizer-english-NOUVEAU
& target/debug/bailey-core.exe curate-text `
  --data runs/tokenizer-english-NOUVEAU/datasets/cycle-0001 `
  --out data/wikipedia-en-tokenizer-curated-NOUVEAU
```

Résultat conservé : `runs/tokenizer-english-20260927-v1` et
`data/wikipedia-en-tokenizer-curated-v1`. Aucun passage commun significatif n'a
été retiré. La normalisation des séparateurs conserve 935 307 octets train,
167 727 validation et 30 404 test. Les partitions de validation/test anglaises
ne sont pas utilisées pour former le tokenizer.

## Quotas et traçabilité

La configuration `mix.json` contient la graine, un budget total de texte,
les parts `[70,15,15]` et chaque source avec catégorie, partition, chemin,
SHA256 et manifeste de provenance. Les chemins relatifs sont résolus depuis
le dossier de configuration.

Les parts sont mesurées en **octets UTF-8 sources**, pas en documents, mots ou
tokens. Des fragments d'au plus 4 096 octets, coupés aux frontières UTF-8, sont
tirés sans remise. La catégorie la moins fournie limite le volume total : il
n'y a ni répétition pour remplir un quota, ni ajout artificiel de fusions BPE.
Un séparateur de ligne est ajouté après chaque fragment ; ses octets sont
comptés séparément et réservés dans le plafond total.

Le manifeste final conserve la configuration, la provenance, les offsets et
longueurs des fragments, les parts réellement obtenues et le SHA256 de train.txt.
Le mélange ne crée que train.txt : les sondes de développement du tokenizer sont
stockées à part dans `assets/tokenizer/probes-v1.json` et ne sont pas des sources.

Le tokenizer exporté conserve aussi une copie du manifeste source, son empreinte
et celle de tokenizer.json. Un `train_sha256` périmé est refusé avant BPE. Le
statut reste `candidate_not_frozen` ; aucune compétence linguistique n'est déduite
de la taille du vocabulaire.

## Limites

Ce mélange pilote lit au plus 512 Mio de sources en RAM et produit au plus
256 Mio de texte. La déduplication intégrée retire les fichiers entiers identiques,
pas toutes les répétitions entre fragments ni les documents proches. Les fragments
peuvent couper une phrase ou un programme ; ce corpus est destiné au découpage
du tokenizer, pas directement à une leçon de code.

Les commentaires et la documentation des sources de code comptent dans la
catégorie code : les ratios désignent les sources, pas une analyse linguistique
de chaque caractère. Les textes restent dominés par l'encyclopédie et les sources
Rust ; la diversité doit être élargie avant le gel définitif.
