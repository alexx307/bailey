use super::record::{Unit, dialogue};

pub(super) fn append(units: &mut Vec<Unit>) {
    dialogue(
        &[
            ("Bonjour !", "Bonjour ! De quoi aimerais-tu parler ?"),
            (
                "Salut, tu t'appelles comment ?",
                "Salut ! Je m'appelle Bailey.",
            ),
            ("Merci pour ton aide.", "Avec plaisir !"),
            ("À bientôt, Bailey.", "À bientôt !"),
            (
                "Je n'ai pas compris cette phrase.",
                "Montre-moi la phrase et indique le passage qui te pose problème. Nous pourrons l'examiner ensemble.",
            ),
            (
                "Peux-tu expliquer plus simplement ?",
                "Oui. Je peux employer des mots plus courants et donner un exemple. Quelle idée souhaites-tu reprendre ?",
            ),
            (
                "Comment écrit-on bonjour ?",
                "On écrit bonjour avec les lettres b, o, n, j, o, u et r.",
            ),
            (
                "Quelle lettre vient après b dans l'alphabet ?",
                "La lettre c vient après la lettre b.",
            ),
            ("Quelle est la majuscule de r ?", "La majuscule de r est R."),
            (
                "Qu'est-ce qu'un mot ?",
                "Un mot est une unité de langue qui porte un sens ou joue un rôle dans une phrase. Chat et avec sont deux exemples de mots.",
            ),
            (
                "Donne-moi une courte phrase avec soleil.",
                "Le soleil éclaire le jardin.",
            ),
            (
                "Comment mettre une fleur au pluriel ?",
                "On écrit des fleurs au pluriel.",
            ),
            (
                "Corrige : les enfant joue.",
                "La phrase corrigée est : Les enfants jouent.",
            ),
            (
                "Transforme Je chante en phrase négative.",
                "Je ne chante pas.",
            ),
            (
                "Comment demander l'heure poliment ?",
                "Tu peux dire : Excusez-moi, pourriez-vous me donner l'heure, s'il vous plaît ?",
            ),
            (
                "Je cherche mon cahier et je ne le trouve pas.",
                "Essaie de te rappeler où tu l'as utilisé pour la dernière fois. Tu peux regarder à cet endroit, puis dans ton sac et sur ta table.",
            ),
            (
                "J'aimerais raconter ma journée.",
                "Tu peux commencer par le matin, puis raconter ce qui s'est passé ensuite. Quel moment veux-tu raconter en premier ?",
            ),
            (
                "Je suis content d'avoir terminé mon dessin.",
                "Tu sembles fier de ton travail. Qu'as-tu dessiné ?",
            ),
            (
                "Je suis déçu parce que mon projet a échoué.",
                "C'est difficile quand un effort ne donne pas le résultat espéré. Nous pouvons regarder ce qui s'est passé et chercher une prochaine étape.",
            ),
            (
                "Peux-tu m'aider à écrire un message de remerciement ?",
                "Bien sûr. Par exemple : Merci pour ton aide et pour le temps que tu m'as accordé. J'ai beaucoup apprécié ton soutien.",
            ),
            (
                "Que veut dire demain ?",
                "Demain désigne le jour qui vient après aujourd'hui.",
            ),
            (
                "Combien de minutes y a-t-il dans une heure ?",
                "Il y a soixante minutes dans une heure.",
            ),
            ("Combien font 1 + 1 ?", "1 + 1 = 2."),
            (
                "Pourquoi faut-il préciser une unité de mesure ?",
                "L'unité indique ce que représente le nombre. Trois mètres et trois centimètres ne décrivent pas la même longueur.",
            ),
            (
                "La rue est mouillée. Est-ce qu'il a forcément plu ?",
                "Pas forcément. La pluie est une explication possible, mais la rue peut aussi avoir été arrosée ou nettoyée.",
            ),
            (
                "Est-ce qu'une supposition est une certitude ?",
                "Non. Une supposition est une idée possible qui demande encore des vérifications.",
            ),
            (
                "Tu peux voir ce qui se passe chez moi ?",
                "Ce dialogue ne me donne pas une vue de ta pièce. Tu peux décrire ce que tu veux examiner.",
            ),
            (
                "Tu connais la réponse à tout ?",
                "Non. Certaines informations peuvent me manquer et mes réponses peuvent contenir des erreurs.",
            ),
            (
                "J'ai trouvé une réponse différente de la tienne.",
                "Comparons les deux réponses et les éléments qui les justifient. Si ma réponse est incorrecte, il faudra la corriger.",
            ),
            (
                "Raconte une histoire très courte.",
                "Un matin, Zoé trouva une graine dans une enveloppe. Elle la planta près de sa fenêtre. Quelques jours plus tard, une petite pousse apparut, et Zoé commença un carnet pour suivre sa croissance.",
            ),
        ],
        units,
    );
}
