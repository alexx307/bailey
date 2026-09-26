# Premier cours de dialogue français

170 échanges d'entraînement originaux, rédigés et relus par l'assistant de
développement le 26 septembre 2026 ; aucun texte externe ou poids préentraîné.
Quatre sujets : contact (45), langue (52), quotidien (43), nombres (30).
28 reformulations de validation servent au développement ; 28 autres restent
réservées au test final. Les formulations proches sont intentionnelles : ce cours
mesure un transfert limité entre formulations, pas des connaissances nouvelles.

Les données sources sont versionnées ici. `prepare-dialogue` contrôle les
questions vides, marqueurs réservés et doublons exacts normalisés, puis exporte
les trois partitions et leurs empreintes dans un nouveau dossier sous `data/`.
La préparation lit le test pour contrôler la séparation ; ni l'entraîneur ni
la sélection des checkpoints ne le lisent.

Le mode `--objective dialogue` présente les échanges séparément et supervise
seulement les tokens de réponse, y compris leur fin ; questions et remplissage
du lot sont exclus de la perte. Les échanges trop longs sont rejetés explicitement.
Le mode `next-token` reste disponible pour les corpus de texte continu.
Le tokenizer précédent est conservé pour préserver la compatibilité des poids.

Le cours reste minuscule : une réponse correcte à un exemple vu peut être une
mémorisation. Il ne rend pas le modèle généraliste et ne garantit pas l'arithmétique.
