# Conventions de Bailey

Ces règles viennent des demandes explicites du propriétaire du projet.

## Direction actuelle (26 septembre 2026)

- Construire d'abord une fondation de langage généraliste, en priorité française.
  Le code devient une spécialisation ultérieure, après des bases de langage
  évaluées. Ne pas reprendre l'ancien objectif « modèle de code d'abord ».
- Le projet actif est le workspace Rust et `crates/bailey-core/`, avec Candle.
  L'ancien package Burn et le `src/` racine ont été supprimés à la demande du
  propriétaire. Les modules Internet utiles ont été transférés dans le cœur.
- Baseline : environ 100 M de paramètres, GQA, RoPE, RMSNorm, SwiGLU,
  embeddings d'entrée/sortie partagés. Le tokenizer BPE est appris localement.
- Une architecture ou un tokenizer incompatibles exigent leur propre première
  initialisation. Ne jamais prétendre convertir automatiquement les anciens
  poids. Ensuite reprendre les checkpoints, sans réinitialiser les acquis.
- Valider causalité, gradients de chaque paramètre, mise à jour, sauvegarde et
  rechargement avant un long entraînement. Les opérations d'inférence optimisées
  sans backward ne conviennent pas à l'entraînement.
- Distinguer corpus de démonstration, pré-entraînement français substantiel et
  apprentissage du dialogue. Quelques centaines de leçons ne suffisent pas au
  pré-entraînement d'un modèle généraliste de 100 M de paramètres.
- MoE, adaptateurs, BitNet, quantification et auto-évolution sont des expériences
  ultérieures : les ajouter séparément après mesure d'une baseline fonctionnelle.
- Conserver les anciennes expériences et les archives hors du code actif.

- Construire notre modèle en Rust, sans substituer un modèle préentraîné externe.
  Les poids aléatoires ne servent qu'à la première initialisation. Reprendre les
  poids appris pour les leçons suivantes ; enseigner avec des exemples rédigés et
  vérifiés, notamment le français avant d'élargir les compétences.
- Écrire l'application et l'entraînement en Rust.
- Organiser les dossiers par sujet et sous-sujet, puis les fichiers par
  responsabilité autonome. Ne pas accumuler toute la logique dans `main.rs`.
- Viser au maximum environ 300 lignes par fichier. Découper par responsabilité,
  sans multiplier les fichiers artificiellement pour chaque petite expression.
- Avancer de façon autonome sur les étapes réversibles déjà autorisées.
- Déléguer les travaux indépendants à des sous-agents : responsabilité et fichiers
  attribués clairement, intégration et vérification par l'agent principal.
  Si une limite technique empêche la délégation, l'indiquer et poursuivre.
- Garder le plan et les résultats vérifiés à jour dans `docs/`.
- Distinguer entraînement des poids, mémoire de l'agent et modification de code.
- Ne jamais présenter une baisse de perte comme une preuve de compétence de code.
- Réserver des données de test hors entraînement et hors sélection des versions.
- Vérifier les modifications avec formatage, analyse statique et tests adaptés.
- Conserver les versions précédentes ; ne pas écraser un entraînement existant.
