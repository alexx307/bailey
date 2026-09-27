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
- Priorité après les sorties répétitives : diagnostics des tokens, exemples
  français vérifiés et évaluations de générations sur validation. Le sampling
  ou une pénalité de répétition ne constituent pas un apprentissage.
- Pour le dialogue supervisé, conserver chaque échange complet et masquer les
  questions et le remplissage dans la perte ; apprendre le marqueur de fin.
  Évaluer sur des échanges indépendants. Ne pas comparer directement cette perte
  avec celle du pré-entraînement sur des fenêtres de texte continu.
- Pour les futures séances cloud, préparer et vérifier les données localement
  avant la location ; suivre `docs/plan/forge.md`. Un benchmark utilisé pour
  promouvoir des candidates est une validation, pas le test final réservé.
- Distinguer le vocabulaire réel du tokenizer, la capacité de la tête du modèle
  et le nombre de tokens du corpus. Ne pas remplacer le tokenizer d'un modèle
  appris sans migration explicite ou expérience indépendante.
- Les shards Forge sont immuables : vérifier les SHA256, la provenance et les
  IDs du tokenizer avant leur utilisation. Ne charger que train et validation
  pendant l'entraînement. Le prototype actuel charge une partition en RAM ; ne
  pas le présenter comme un lecteur en flux adapté à des milliards de tokens.
- `--init-from` ouvre une nouvelle séance avec les meilleurs poids et un nouvel
  AdamW. `resume` continue une séance interrompue depuis `latest.json`, avec les
  moments Adam, le pas global, le planning et les tirages par étape. Toujours
  écrire dans un nouveau dossier ; refuser des données ou réglages modifiés.
- Conserver `best.json` pour la sélection d'inférence et `latest.json` pour la
  reprise. Publier ce dernier seulement après une sauvegarde complète. La reprise
  CPU est testée à l'identique ; ne pas promettre une identité numérique entre
  versions du backend ou matériels différents.
- Décontaminer les corpus avant les expériences : les passages réservés restent
  réservés, les copies sont retirées de l'entraînement. Conserver l'original et
  l'audit des transformations. Cette comparaison de données ne doit pas devenir
  une évaluation du modèle sur le test ni un moyen d'inventer ses réponses.
- Un import incomplet ou interrompu ne produit pas de manifeste consommable par
  Forge. Respecter les budgets et erreurs HTTP ; ne pas intégrer automatiquement
  des textes Internet comme des faits validés.
- Direction tokenizer du 27 septembre : un vocabulaire Byte-Level BPE commun
  pour français, anglais, code et symboles. Comparer des candidats séparés avant
  le gel ; ne pas router entre plusieurs tokenizers dans le modèle actuel.
- Le mélange 70/15/15 est une hypothèse mesurée en octets UTF-8, pas un optimum
  établi. Utiliser uniquement des sources train explicites, avec empreintes et
  provenance. Ne pas répéter un petit corpus pour atteindre artificiellement
  un quota. Les sondes de développement du tokenizer restent hors mélange.
- Un audit de restitution et de fragmentation ne prouve aucune compétence de
  langue ou de code. Le statut `candidate_not_frozen` demeure tant qu'une revue
  sur un corpus représentatif n'a pas justifié le gel. Conserver les poids actifs
  avec leur ancien tokenizer ; aucun remplacement silencieux des IDs.
- Le repère de 20 tokens par paramètre sert à la planification, sans promesse de
  compétence. Distinguer tokens du corpus et occurrences rejouées à l'entraînement.

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
