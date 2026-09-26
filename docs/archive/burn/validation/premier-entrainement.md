# Première validation locale — 22 septembre 2026

## Configuration

- Windows, Intel Core i7-10700K, 32 Go de RAM.
- NVIDIA RTX 5070 Ti, 16 Go de mémoire vidéo, pilote 616.92.
- Rust 1.92.0, Burn 0.21.0 ; WGPU pour le GPU et NdArray pour le CPU.
- Profil Cargo `dev` avec optimisation de niveau 1.

La version Rust est fixée dans le projet, sans changer la version globale
par défaut. Aucun poids préentraîné n'a été chargé.

## Vérifications réussies

- `check-device` : calcul GPU et rétropropagation, somme des carrés = 30,
  gradient = [2, 4, 6, 8].
- `cargo fmt --check` et `cargo clippy --all-targets -- -D warnings`.
- `cargo test --no-default-features` : 7 tests réussis.
- Le contrôle de taille des fichiers Rust dans `tools/quality/check.ps1`.
- Compilation comme bibliothèques des trois corpus Rust avec `rustc`.
- Sauvegarde GPU et rechargement CPU pour évaluer et générer.
- Refus d'écraser un corpus ou une expérience existante.
- Continuation depuis des poids sauvegardés, dans une nouvelle expérience.

Le test de causalité a détecté un masque inversé pendant le développement.
Le masque a été corrigé avant les entraînements ci-dessous et le test passe.

## Expérience principale

```powershell
cargo run -- self-train --out runs/bailey-v0 --rounds 2 --steps 200 --eval-every 50
```

La commande a été exécutée avec le binaire déjà compilé, sans recompilation
dans le temps mesuré. Les fichiers locaux sont dans `runs/bailey-v0/`.
Ne pas réutiliser ce nom pour un nouvel entraînement : il existe déjà.

| Mesure | Résultat |
| --- | --- |
| Paramètres | 478 976 |
| Contexte | 128 octets |
| Lots | 8 fenêtres par étape |
| Cycles | 2 × 200 étapes |
| Perte de validation initiale | 5,6314 |
| Perte de validation finale | 0,3382 |
| Temps de la boucle, évaluations et sauvegardes comprises | 12,60 s |
| Perte sur le jeu de test après rechargement CPU | 0,3637 |

La validation utilise 16 fenêtres fixes, le test 32 fenêtres fixes.
Ces mesures ne sont pas des scores de résolution de problèmes. Le corpus
ne contient que 2 048 exemples issus de huit patrons simples.

Un essai préalable de 137 216 paramètres, 60 étapes et un contexte de 64 octets
a également fonctionné : perte de validation de 5,7170 à 2,7165 en 4,8 s.
Ces durées ne se transposent pas à de grands modèles ou corpus.

## Génération observée et prochaine limite

Une génération depuis `pub fn `, avec température 0, a produit notamment :

```text
pub fn even_116(x: i64) -> bol {
```

Le type `bol` est invalide. La suite comportait également des erreurs.
La génération et le rechargement fonctionnent, mais cette version ne produit
pas encore du Rust fiable. Rien de ce qu'elle a généré n'a été exécuté.

La prochaine étape du plan est de travailler sur des exercices demande/solution
et leur évaluation fonctionnelle indépendante. Réduire davantage la perte sur
ces seuls patrons ne suffirait pas à démontrer une compétence générale.

## Délégation

Deux sous-agents ont été affectés au modèle et aux données. Ils ont apporté
des éléments de cadrage, puis leurs tâches ont été interrompues par une limite
d'usage. L'agent principal a réalisé et vérifié l'implémentation finale.
