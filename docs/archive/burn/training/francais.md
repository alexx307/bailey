# Apprendre les bases du français

Le modèle est celui de Bailey, avec ses poids déjà entraînés sur le code. Le
corpus français est rédigé par les assistants et produit par Rust : salutations,
identité, alphabet, lettres, vocabulaire, grammaire et calculs élémentaires. Cela
transmet des exemples d'enseignement, pas les paramètres internes des assistants.

## Séance visible

```powershell
.\tools\launch\start-french.ps1 -Train
```

La séance prépare `data/french-foundation`, mesure les réponses avant/après,
effectue 6 000 étapes puis ouvre la console. Elle reprend la dernière version
française indiquée dans `runs/french-active.json` si elle existe, sinon la séance
Rust initiale. `-ModelRun chemin` choisit explicitement une version. Chaque
entraînement conserve son propre dossier et ses checkpoints ; les poids ne
repartent pas au hasard. L'optimiseur est recréé.

Pour ouvrir la dernière séance française sans nouvel entraînement :

```powershell
.\tools\launch\start-french.ps1
```

Écris `salut`, `1 + 1 =`, `Récite l'alphabet.` ou une autre question courte.
`/dialogue` utilise le format des leçons et un marqueur de fin de réponse.
`/brut` permet de revenir à la complétion libre. Les réponses viennent des poids
du modèle ; aucun dictionnaire de réponses ou calculateur ne les remplace.

## Ce qui est évalué

899 exemples sont répartis avant entraînement. Le français réserve des
reformulations tout en partageant les faits connus. Les calculs réservent des
paires numériques, y compris l'ordre inversé. Les manifestes donnent les détails.
La validation sert à sélectionner les poids. Le jeu de test doit rester réservé
au bilan final. Les générations s'arrêtent au marqueur appris ou à 180 octets,
sans connaître la longueur de la réponse attendue.

Le score exact pénalise aussi des variantes linguistiques correctes : il faut
lire les réponses et ne pas assimiler toute différence à une erreur. Le contexte
actuel est de 128 octets ; chaque question est indépendante et doit être courte.
Ce modèle de 478 976 paramètres ne constitue pas un assistant généraliste. La
mémorisation de salutations ou d'opérations ne démontre pas un raisonnement fiable.

## Internet et apprentissage continu

`/recherche sujet` enregistre des articles Wikipédia avec leur provenance.
`/memoire sujet` consulte les pages déjà conservées. La
[boucle Internet](../learning/internet.md) peut ensuite entraîner des candidates
en rejouant les leçons françaises et en mesurant leur conservation.

Les sujets ne sont pas limités par une liste de thèmes. L'application ne possède
pas de mécanisme fiable pour déterminer seule qu'une réponse est fausse ou qu'un
sujet est maîtrisé. Les sujets d'étude sont donc explicites et configurables.
