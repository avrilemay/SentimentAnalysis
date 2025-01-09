"""
8_tdidf.py

Ce script utilise tf-idf pour extraire les termes les plus significatifs et
génère des nuages de mots pour visualiser les sentiments positifs et négatifs
sur différentes périodes (mensuelle, annuelle et globale).

Fonctionnalités :
- connexion à une base de données postgresql
- analyse des posts et des commentaires
- filtrage par catégories grammaticales
- génération de nuages de mots pour visualisation

Exemple d'utilisation :
    python 8_tdidf.py --tables table1 table2 --analyze_posts
    --analyze_comments --pos noun adj adv --output_dir ../sorties/plots
"""
from random import randint  # nombres aléatoires
import pandas as pd  # manipulation de données
from sklearn.feature_extraction.text import TfidfVectorizer  #  TfidfVectorizer
from wordcloud import WordCloud  # nuages de mots
import matplotlib; matplotlib.use('Agg')   # backend non interactif "Agg"
import matplotlib.pyplot as plt
import argparse  # arguments en ligne de commande
from sqlalchemy import create_engine  # connexion BD
import sys  # paramètres système [mode test]
import spacy  # traitement du langage naturel
import os  # gestion des dossiers

# charge le modèle spacy pour l'analyse grammaticale
nlp = spacy.load("en_core_web_sm")  # modèle linguistique anglais


def red_shades_color_func(word, font_size, position, orientation,
                          random_state=None, **kwargs):
    """
    Définit des nuances de rouge pour les termes négatifs.

    Arguments :
        word (str): terme
        font_size (int): taille de la police
        position (tuple): position du mot
        orientation (str): orientation du mot
        random_state: état aléatoire
        **kwargs: autres arguments

    Retourne :
        str: code couleur rgb
    """
    # retourne rouge avec des variations aléatoires
    return (f"rgb({randint(150, 255)}, " # couleur rouge aléatoire
            f"{randint(0, 50)}, {randint(0, 50)})")


def get_sqlalchemy_engine():
    """
    Crée un moteur sqlalchemy pour se connecter à postgresql.

    Retourne :
        sqlalchemy engine: moteur sqlalchemy configuré
    """
    # créé et retourne le moteur de connexion à la base de données PostgreSQL
    return create_engine(
        "postgresql://avrile:projet@localhost:5432/projet_reddit"
    )  # spécifie la chaîne de connexion


def main(tables, analyze_posts, analyze_comments, pos_filters, output_dir):
    """
    Fonction principale pour l'analyse des sentiments.

    Arguments :
        tables (list): liste des noms de tables à analyser
        analyze_posts (bool): indique si les posts doivent être analysés
        analyze_comments (bool): indique si les commentaires doivent être analysés
        pos_filters (list): liste des catégories grammaticales à filtrer
        output_dir (str): dossier où sauvegarder les graphiques
    """
    # obtenir le moteur SQL pour la connexion à la base de données
    engine = get_sqlalchemy_engine()  # appeler la fonction pour l'engine SQL

    # initialise les dictionnaires pour stocker les termes positifs et négatifs
    positive_terms_month = {}  # termes positifs mensuels
    negative_terms_month = {}  # termes négatifs mensuels
    positive_terms_year = {}  # termes positifs annuels
    negative_terms_year = {}  # termes négatifs annuels
    positive_terms_all = {}  # termes positifs globaux
    negative_terms_all = {}  # termes négatifs globaux

    excluded_words = {}  # initialise les mots exclus, vide

    # parcourt chaque table fournie en argument
    for table in tables:
        # construit la requête SQL pour sélectionner toutes les données
        query = f"SELECT * FROM {table}"  # crée la requête SQL
        # lit les données de la table dans un dataframe pandas
        data = pd.read_sql(query, engine)  # charge les données

        # analyse les posts si demandé
        if analyze_posts:
            # extrait les termes positifs et négatifs des posts
            positive_terms, negative_terms = analyze_data(
                data, 'lower', pos_filters, excluded_words
            )  # appelle analyze_data pour les posts
            # met à jour les termes globaux
            update_terms(positive_terms_all, negative_terms_all, positive_terms,
                         negative_terms)  # met à jour les dicts globaux

            if 'month' in table:  # vérifie si la table est mensuelle
                # met à jour les termes mensuels
                update_terms(positive_terms_month, negative_terms_month,
                             positive_terms, negative_terms)
            elif 'year' in table:  # vérifie si la table est annuelle
                # met à jour les termes annuels
                update_terms(positive_terms_year, negative_terms_year,
                             positive_terms, negative_terms)

        # analyse les commentaires si demandé
        if analyze_comments:
            # extrait les termes positifs et négatifs des commentaires
            positive_terms, negative_terms = analyze_data(
                data, 'lower', pos_filters, excluded_words
            )  # appelle analyze_data pour les commentaires
            # met à jour les termes globaux
            update_terms(positive_terms_all, negative_terms_all, positive_terms,
                         negative_terms)  # met à jour les dicts globaux

            if 'month' in table:  # vérifie si la table est mensuelle
                # met à jour les termes mensuels
                update_terms(positive_terms_month, negative_terms_month,
                             positive_terms, negative_terms)
            elif 'year' in table:  # vérifie si la table est annuelle
                # met à jour les termes annuels
                update_terms(positive_terms_year, negative_terms_year,
                             positive_terms, negative_terms)

    # ne pas avoir les mêmes mots dans positifs et négatifs all time
    common_words_all = set(positive_terms_all.keys()) & set(
        negative_terms_all.keys())   # identifie les mots communs
    for word in common_words_all:       # récupération des scores + et -
        pos_score = positive_terms_all[word]
        neg_score = negative_terms_all[word]
        # conserve le mot uniquement là où le score est le plus fort
        if pos_score > neg_score:
            del negative_terms_all[word]
        else:
            del positive_terms_all[word]

    # ne pas avoir les mêmes mots dans positifs et négatifs mensuels
    common_words_month = set(positive_terms_month.keys()) & set(
        negative_terms_month.keys())  # les mots communs dans le set
    for word in common_words_month:   # récupération des scores
        pos_score = positive_terms_month[word]
        neg_score = negative_terms_month[word]
        if pos_score > neg_score:   # on conserve uniquement le score fort
            del negative_terms_month[word]
        else:
            del positive_terms_month[word]

    # ne pas avoir les mêmes mots dans positifs et négatifs annuels
    common_words_year = set(positive_terms_year.keys()) & set(
        negative_terms_year.keys())    # les mots communs
    for word in common_words_year:   # récupération scores + et -
        pos_score = positive_terms_year[word]
        neg_score = negative_terms_year[word]
        if pos_score > neg_score:   # selon le score le plus élevé
            del negative_terms_year[word]   # on ne garde le mot qu'à 1 endroit
        else:
            del positive_terms_year[word]

    # générer les nuages de mots pour les termes collectés
    generate_wordcloud(
        positive_terms_month,  # termes positifs mensuels
        "nuage de mots positifs - mensuel",  # titre du nuage
        "positive",  # type de sentiment
        "positive_month.png",  # nom du fichier
        output_dir  # dossier de sortie
    )  # générer le nuage positif mensuel

    generate_wordcloud(
        negative_terms_month,  # termes négatifs mensuels
        "nuage de mots négatifs - mensuel",  # titre du nuage
        "negative",  # type de sentiment
        "negative_month.png",  # nom du fichier
        output_dir  # dossier de sortie
    )  # générer le nuage négatif mensuel

    generate_wordcloud(
        positive_terms_year,  # termes positifs annuels
        "nuage de mots positifs - annuel",  # titre du nuage
        "positive",  # type de sentiment
        "positive_year.png",  # nom du fichier
        output_dir  # dossier de sortie
    )  # générer le nuage positif annuel

    generate_wordcloud(
        negative_terms_year,  # termes négatifs annuels
        "nuage de mots négatifs - annuel",  # titre du nuage
        "negative",  # type de sentiment
        "negative_year.png",  # nom du fichier
        output_dir  # dossier de sortie
    )  # générer le nuage négatif annuel

    generate_wordcloud(
        positive_terms_all,  # termes positifs globaux
        "nuage de mots positifs - global",  # titre du nuage
        "positive",  # type de sentiment
        "positive_all.png",  # nom du fichier
        output_dir  # dossier de sortie
    )  # générer le nuage positif global

    generate_wordcloud(
        negative_terms_all,  # termes négatifs globaux
        "nuage de mots négatifs - global",  # titre du nuage
        "negative",  # type de sentiment
        "negative_all.png",  # nom du fichier
        output_dir  # dossier de sortie
    )  # générer le nuage négatif global


    # message final après génération des nuages de mots
    print("Les nuages de mots ont été générés et enregistrés dans le répertoire "
          "passé en argument\n\n")




def update_terms(positive_terms, negative_terms,
                 new_positive_terms, new_negative_terms):
    """
    Met à jour les dictionnaires de termes positifs et négatifs

    Arguments :
        positive_terms (dict): termes positifs existants
        negative_terms (dict): termes négatifs existants
        new_positive_terms (dict): nouveaux termes positifs à ajouter
        new_negative_terms (dict): nouveaux termes négatifs à ajouter
    """
    # ajouter ou mettre à jour les mots positifs
    for word, score in new_positive_terms.items():  # parcourir nouveaux +
        if word in positive_terms:  # vérifier si le mot existe déjà
            positive_terms[word] += score  # ajouter le score existant
        else:
            positive_terms[word] = score  # ajouter le nouveau mot

    # ajouter ou mettre à jour les mots négatifs
    for word, score in new_negative_terms.items():  # parcourir nouveaux -
        if word in negative_terms:  # vérifier si le mot existe déjà
            negative_terms[word] += score  # ajouter le score existant
        else:
            negative_terms[word] = score  # ajouter le nouveau mot


def analyze_data(data, text_column, pos_filters, excluded_words):
    """
    Analyse les données textuelles pour extraire les termes principalement
    positifs et négatifs

    Arguments :
        data (pd.DataFrame): données à analyser
        text_column (str): nom de la colonne texte
        pos_filters (list): filtres de catégories grammaticales
        excluded_words (dict): mots à exclure des analyses

    Retourne :
        tuple: (dict, dict) termes principalement positifs et négatifs
    """
    # définit des mots à exclure personnalisés
    custom_stop_words = ['www', 'http']  # ajoute des stop words personnalisés
    # combine les stop words de sklearn avec les personnalisés
    combined_stop_words = list(
        TfidfVectorizer(stop_words='english').get_stop_words()
    ) + custom_stop_words  # combine les listes de stop words

    # extrait et nettoyer les textes
    texts = data[text_column].fillna('').str.strip()  # remplace NAN et strip
    non_empty_texts = texts[texts != '']  # filtre les textes non vides

    # vérifie s'il y a des textes à analyser
    if not non_empty_texts.empty:
        filtered_texts = []  # initialise la liste des textes filtrés
        # traite les textes avec spacy en lots
        for doc in nlp.pipe(non_empty_texts, batch_size=50):
            if 'all' in pos_filters:
                # garde tous les tokens
                filtered_tokens = [token.text for token in doc]
            else:
                # filtre les tokens selon les POS spécifiés
                filtered_tokens = [
                    token.text for token in doc if
                    token.pos_ in pos_filters
                ]           # filtre par catégorie grammaticale
            # joint les tokens filtrés en une chaîne
            filtered_texts.append(" ".join(filtered_tokens))

        # initialise le vectorizer tf-idf
        vectorizer = TfidfVectorizer(
            stop_words=combined_stop_words,  # définit les stop words
            ngram_range=(1, 1),  # utilise des unigrammes
            max_features=100  # limite à 100 caractéristiques (=mots)
        )
        # calcule la matrice tf-idf
        tfidf_matrix = vectorizer.fit_transform(filtered_texts)  # tf-idf
        feature_names = vectorizer.get_feature_names_out()  # obtient les mots
        scores = tfidf_matrix.sum(axis=0).A1  # somme les scores tf-idf
        tfidf_scores = list(zip(feature_names, scores))  # associe mots/scores
        # trie les scores par ordre décroissant
        tfidf_scores_sorted = sorted(
            tfidf_scores, key=lambda x: x[1], reverse=True
        )  # trie les scores

        # filtre les termes avec un seuil de score
        filtered_tfidf_scores = [
            (word, score) for word, score in tfidf_scores_sorted
            if score > 0.1 and word not in excluded_words
        ]  # applique le seuil de filtrage

        # sépare les données en positives et négatives
        positive_data = data[
            data['lower_zero-shot'].str.contains('positive',
                                                 case=False, na=False)
        ]  # sélectionne les données positives
        negative_data = data[
            data['lower_zero-shot'].str.contains('negative',
                                                 case=False, na=False)
        ]  # sélectionne les données négatives

        # extrait les textes positifs et négatifs
        positive_texts = positive_data[text_column].fillna('').str.strip()
        negative_texts = negative_data[text_column].fillna('').str.strip()

        # filtre les textes positifs selon les POS
        positive_filtered_texts = []  # initialise la liste
        # analyse les textes positifs:
        for doc in nlp.pipe(positive_texts, batch_size=50):
            if 'all' in pos_filters:  # vérifie si tous les POS sont inclus
                # garde tous les tokens:
                filtered_tokens = [token.text for token in doc]
            else:
                # filtre les tokens selon les POS spécifiés
                filtered_tokens = [
                    token.text for token in doc if token.pos_ in pos_filters
                ]  # filtre par catégorie grammaticale
            # joint les tokens filtrés en une chaîne
            positive_filtered_texts.append(" ".join(filtered_tokens))

        # filtre les textes négatifs selon les POS
        negative_filtered_texts = []  # initialise la liste
        for doc in nlp.pipe(negative_texts, batch_size=50):
            if 'all' in pos_filters:  # vérifie si tous les POS sont inclus
                filtered_tokens = [token.text for token in doc]
            else:
                # filtre les tokens selon les POS spécifiés
                filtered_tokens = [
                    token.text for token in doc if token.pos_ in pos_filters
                ]  # filtre par catégorie grammaticale
            # joint les tokens filtrés en une chaîne
            negative_filtered_texts.append(" ".join(filtered_tokens))

        # initialise le vectorizer tf-idf pour les positifs
        positive_vectorizer = TfidfVectorizer(
            stop_words=combined_stop_words,  # définit les stop words
            ngram_range=(1, 1),  # utilise des unigrammes
            max_features=100  # limite à 100 caractéristiques
        )
        # initialise le vectorizer tf-idf pour les négatifs
        negative_vectorizer = TfidfVectorizer(
            stop_words=combined_stop_words,  # définit les stop words
            ngram_range=(1, 1),  # utilise des unigrammes
            max_features=100  # limite à 100 caractéristiques
        )

        # calcule tf-idf pour les textes positifs
        positive_tfidf_matrix = positive_vectorizer.fit_transform(
            positive_filtered_texts
        )  # applique tf-idf aux textes positifs
        # calcule tf-idf pour les textes négatifs
        negative_tfidf_matrix = negative_vectorizer.fit_transform(
            negative_filtered_texts
        )  # applique tf-idf aux textes négatifs

        # obtient les mots des positifs
        positive_feature_names = positive_vectorizer.get_feature_names_out()
        # obtient les mots des négatifs
        negative_feature_names = negative_vectorizer.get_feature_names_out()

        # somme les scores positifs
        positive_scores = positive_tfidf_matrix.sum(axis=0).A1
        # somme les scores négatifs
        negative_scores = negative_tfidf_matrix.sum(axis=0).A1

        # créé un dictionnaire pour les scores positifs
        positive_scores_dict = {
            word: score for word, score in zip(
                positive_feature_names, positive_scores
            )  # associe mots et scores positifs
        }
        # créé un dictionnaire pour les scores négatifs
        negative_scores_dict = {
            word: score for word, score in zip(
                negative_feature_names, negative_scores
            )
        }  # associe mots et scores négatifs

        # extrait les termes principalement positifs
        mostly_positive_terms = {
            word: score for word, score in positive_scores_dict.items()
            if (
                word not in negative_scores_dict or
                score > 1.5 * negative_scores_dict.get(word, 0)
            ) and word not in excluded_words
        }  # filtre les termes positifs dominants

        # extrait les termes principalement négatifs
        mostly_negative_terms = {
            word: score for word, score in negative_scores_dict.items()
            if (
                word not in positive_scores_dict or
                score > 1.5 * positive_scores_dict.get(word, 0)
            ) and word not in excluded_words
        }  # filtre les termes négatifs dominants

        # assure qu'aucun mot n'apparaît dans les deux sets
        common_words = set(mostly_positive_terms.keys()) & set(
            mostly_negative_terms.keys())   # récupère les mots communs
        for word in common_words:   # obtient score + et - pour chaque mot
            pos_score = mostly_positive_terms[word]
            neg_score = mostly_negative_terms[word]
            # conserve le mot dans le dictionnaire où il a le score le + fort
            if pos_score > neg_score:
                del mostly_negative_terms[word]
            else:
                del mostly_positive_terms[word]


        # utilise liste tuples filtered_tfidf_scores pour filtrer les termes
        filtered_words = {w for w, s in filtered_tfidf_scores}  # = ensemble
        # intersection avec les termes positifs
        mostly_positive_terms = {
            w: s for w, s in mostly_positive_terms.items() if w in filtered_words
        }  # filtre les positifs
        # intersection avec les termes négatifs
        mostly_negative_terms = {
            w: s for w, s in mostly_negative_terms.items() if w in filtered_words
        }  # filtre les négatifs

        # retourne les termes positifs et négatifs
        return mostly_positive_terms, mostly_negative_terms

    # si aucun texte à analyser, retourne des dictionnaires vides
    return {}, {}  # renvoie des dictionnaires vides


def generate_wordcloud(term_scores, title, sentiment, filename, output_dir):
    """
    Génère et enregistre un nuage de mots à partir des scores des termes

    Arguments :
        term_scores (dict): dictionnaire des termes et leurs scores
        title (str): titre du graphique
        sentiment (str): sentiment ("positive" ou "negative")
        filename (str): nom du fichier pour sauvegarder le nuage
        output_dir (str): dossier où sauvegarder les graphiques
    """
    # vérifie s'il y a des termes à afficher
    if term_scores:
        # créé le nuage de mots avec une couleur spécifique si négatif
        wordcloud = WordCloud(
            width=800, height=400, background_color='white',
            color_func=red_shades_color_func if sentiment == "negative" else None
        ).generate_from_frequencies(term_scores)  # génère le nuage

        plt.figure(figsize=(10, 5))  # définit la taille de la figure
        plt.imshow(wordcloud, interpolation='bilinear')  # affiche le nuage
        plt.axis('off')  # masque les axes
        plt.title(title, fontsize=16)  # ajoute le titre

        os.makedirs(output_dir, exist_ok=True)  # créé dossier sortie si néces
        plt.savefig(os.path.join(output_dir, filename))  # sauvegarde le nuage
        # plt.show()  # affiche le nuage - désactiver pour comptabilité linux
        plt.close()  # ferme la figure pour libérer la mémoire
    else:
        # affiche un message si pas assez de termes
        print(
            f"pas assez de termes pour générer un nuage de mots ({sentiment})."
        )  # notifie l'absence de termes


if __name__ == "__main__":
    """
    Point d'entrée du script pour la génération des nuages de mots.
    """
    # définit arguments pour l'exécution pendant la phase de développement
    #sys.argv = [
        #'8_tdidf.py',
        #'--tables',
        #'posts_table_month_controversial_11_15',
        #'posts_table_month_top_11_15',
        #'posts_table_year_controversial_11_15',
        #'posts_table_year_top_11_15',
        #'comments_table_month_controversial_11_15',
        #'comments_table_month_top_11_15',
        #'comments_table_year_controversial_11_15',
        #'comments_table_year_top_11_15',
        #'--analyze_posts',
        #'--analyze_comments',
        #'--pos', 'NOUN', 'PROPN', 'ADJ', 'ADV',
        #'--output_dir', '../sorties/tdidf_plots'
    #]

    # configure le parser d'arguments
    parser = argparse.ArgumentParser(
        description='analyse des sentiments avec tf-idf et nuages de mots '
                    'sur des tables de base de données.'
    )  # créé le parser avec une description

    parser.add_argument(
        '--tables', type=str, nargs='+', required=True,
        help='liste des tables à analyser.'
    )  # ajoute l'argument pour les tables

    parser.add_argument(
        '--analyze_posts', action='store_true',
        help='analyser les posts.'
    )  # ajoute l'argument pour analyser les posts

    parser.add_argument(
        '--analyze_comments', action='store_true',
        help='analyser les commentaires.'
    )  # ajoute l'argument pour analyser les commentaires

    parser.add_argument(
        '--pos', type=str, nargs='+', choices=[
            'ADJ', 'NOUN', 'VERB', 'ADV', 'PROPN', 'all'
        ], default=['all'],
        help='catégories grammaticales à analyser : ADJ, NOUN, VERB, '
             'ADV, PROPN, ou all (sans filtre).'
    )  # ajoute l'argument pour les filtres de catégories grammaticales

    parser.add_argument(
        '--output_dir', type=str, default='plot',
        help='dossier où sauvegarder les graphiques.'
    )  # ajoute l'argument pour le dossier de sortie

    # parse les arguments de la ligne de commande
    args = parser.parse_args()  # analyse les arguments fournis

    print("\n\nLa création des nuages de mots est en cours. Cela peut prendre "
          "plusieurs minutes.")   # affiche un message sur le terminal

    # exécute la fonction main avec les arguments parsés
    main(
        args.tables, args.analyze_posts,
        args.analyze_comments, args.pos, args.output_dir
    )  # appelle main avec les arguments analysés
