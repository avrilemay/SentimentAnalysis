"""
2_cleaning.py

Ce script nettoie le texte dans les tables d'une base PostgreSQL en appliquant
des transformations choisies par l'utilisateur (suppression de liens,
ponctuation, etc.) et ajoute une colonne pour le texte nettoyé tout en
conservant les données originales.

Principales fonctionnalités :
- Connexion à PostgreSQL avec SQLAlchemy
- Ajout d'une colonne pour le texte nettoyé si elle n'existe pas
- Nettoyage configurable : liens, ponctuation, majuscules, stopwords, etc
- Mise à jour des tables avec les colonnes nettoyées

Exemple d'utilisation :
    python script.py --tables posts_table comments_table --lowercase
                     --remove_links
"""

import re  # régex pour le nettoyage
import argparse  # gestion des arguments en CLI
from nltk.corpus import stopwords  # liste des mots vides (stopwords)
from sqlalchemy import create_engine, text  # connexion à PostgreSQL
import pandas as pd  # manipulation des données DF
import sys  # gestion des arguments système


def get_sqlalchemy_engine():
    """
    Crée un moteur SQLAlchemy pour PostgreSQL.

    Returns:
        sqlalchemy.engine.Engine: Moteur pour se connecter à la base de données
    """
    # informations de connexion à PostgreSQL
    return create_engine(
        "postgresql://avrile:projet@localhost:5432/projet_reddit"
    )


def clean_text(text, options):
    """
    Nettoie le texte selon les options spécifiées

    Args:
        text (str): Texte à nettoyer
        options (argparse.Namespace): Options de nettoyage

    Returns:
        str: Texte nettoyé.
    """
    # convertir en minuscules
    if options.lowercase:  # option lowercase
        text = text.lower()     # conversion en minuscule

    # supprimer les liens URL
    if options.remove_links:    # option remove_links
        text = re.sub(r'http\S+|www\S+', '', text)
        # remplace URL commençant par http ou www + caractères non espacés

    # supprimer la ponctuation
    if options.remove_punctuation:      # option remove_punctuation
        # remplace tout caractère qui n'est pas un mot (\w) ou un espace (\s)
        text = re.sub(r'[^\w\s]', '', text)

    # supprimer les mots vides
    if options.remove_stopwords:   # option remove_stopwords
        stop_words = set(stopwords.words('english'))   # définit les mots vides
        # joint les mots qui ne sont pas dans la liste des mots vides
        text = ' '.join([word for word in text.split()  # itère sur c/ mot
                        if word not in stop_words])   # condition de filtrage

    # supprimer les nombres
    if options.remove_numbers:      # option remove_numbers
        text = re.sub(r'\d+', '', text)
        # remplace une ou plusieurs chiffres par une chaîne vide

    # corriger les abbréviations
    if options.correct_abbreviations:       # option correct_abbreviations
        abbreviations = {       # dictionnaire des abréviations
            "i'm": "i am", "im": "i am", "you're": "you are",
            "youre": "you are", "he's": "he is", "hes": "he is",
            "she's": "she is", "shes": "she is", "it's": "it is",
            "its": "it is", "we're": "we are", "were": "we are",
            "they're": "they are", "theyre": "they are",
            "can't": "cannot", "cant": "cannot", "won't": "will not",
            "wont": "will not", "don't": "do not", "dont": "do not",
            "doesn't": "does not", "doesnt": "does not",
            "didn't": "did not", "didnt": "did not", "isn't": "is not",
            "isnt": "is not", "aren't": "are not", "arent": "are not",
            "wasn't": "was not", "wasnt": "was not",
            "weren't": "were not", "werent": "were not",
            "haven't": "have not", "havent": "have not",
            "hasn't": "has not", "hasnt": "has not",
            "hadn't": "had not", "hadnt": "had not",
            "couldn't": "could not", "couldnt": "could not",
            "shouldn't": "should not", "shouldnt": "should not",
            "wouldn't": "would not", "wouldnt": "would not",
            "mustn't": "must not", "mustnt": "must not",
            "let's": "let us", "lets": "let us", "that's": "that is",
            "thats": "that is", "who's": "who is", "whos": "who is",
            "what's": "what is", "whats": "what is", "here's": "here is",
            "heres": "here is", "there's": "there is", "theres": "there is",
        }
        # boucle pour remplacer les abbréviations:
        for abbr, full in abbreviations.items():  # itère sur chaque abbr.
            # utilise des limites de mots \b pour correspondance exacte
            pattern = r'\b{}\b'.format(re.escape(abbr))  # construit la regex
            text = re.sub(pattern, full, text)  # remplace l'abbréviation

    if options.remove_extra_spaces:  # supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        # remplace une ou plusieurs espaces blancs par un seul espace et strip
    return text


def main():
    """
    Fonction principale pour exécuter le script de nettoyage dans la bdd
    Ajoute une colonne avec le texte nettoyé en conservant les données brutes
    """
    # parsing des arguments de CLI
    parser = argparse.ArgumentParser(  # créé un parser pour les args
        description='Nettoyer les données textuelles des tables de la base.'
    )
    parser.add_argument('--tables', nargs='+', required=True,
                        help='Liste des tables à nettoyer') # tables à traiter
    parser.add_argument('--lowercase', action='store_true',
                        help='Convertir le texte en minuscules.')  # opt. lower
    parser.add_argument('--remove_links', action='store_true',
                        help='Supprimer les liens.')  # option liens
    parser.add_argument('--remove_punctuation', action='store_true',
                        help='Supprimer la ponctuation.')  # ponctuation
    parser.add_argument('--remove_stopwords', action='store_true',
                        help='Supprimer les mots vides.')   # stopwords
    parser.add_argument('--remove_numbers', action='store_true',
                        help='Supprimer les nombres.')   # numbers
    parser.add_argument('--correct_abbreviations', action='store_true',
                        help='Corriger les abbréviations.')   # abbréviation
    parser.add_argument('--remove_extra_spaces', action='store_true',
                        help='Supprimer les espaces multiples.')  # espaces

    args = parser.parse_args()  # récupère les arguments passés en CLI

    # création d'un suffixe pour le nom de la nouvelle colonne
    suffix = ""   # initialise le suffixe
    if args.lowercase:  # vérifie si lowercase est activé
        suffix += "_lower"  # ajoute _lower au suffixe
    if args.remove_links:  # vérifie si remove_links est activé
        suffix += "_no_url"  # ajoute _no_url au suffixe
    if args.remove_punctuation:  # vérifie si remove_punctuation est activé
        suffix += "_no_punct"  # ajoute _no_punct au suffixe
    if args.remove_stopwords:  # vérifie si remove_stopwords est activé
        suffix += "_no_stopwords"  # ajoute _no_stopwords au suffixe
    if args.remove_numbers:  # vérifie si remove_numbers est activé
        suffix += "_no_numbers"  # ajoute _no_numbers au suffixe
    if args.correct_abbreviations:  # vérifie si correct_abbreviations activé
        suffix += "_no_abbr"  # ajoute _no_abbr au suffixe
    if args.remove_extra_spaces:  # vérifie si remove_extra_spaces est activé
        suffix += "_single_space"  # ajoute _single_space au suffixe

    # si aucune option, utiliser '_raw' (données brutes)
    is_raw = not any([    # définit is_raw
        args.lowercase, args.remove_links, args.remove_punctuation,
        args.remove_stopwords, args.remove_numbers,
        args.correct_abbreviations, args.remove_extra_spaces
    ])
    if is_raw:   # si is_raw est True
        suffix = "_raw"    # définit le suffixe à _raw

    # connexion à la base de données
    engine = get_sqlalchemy_engine()  # obtient l'engine SQLAlchemy

    # pour chaque table spécifiée
    for table_name in args.tables:  # itère sur c/ table et affiche traitement
        print(f"\n\nTraitement de la table {table_name}...")

        # déterminer le type de table
        if 'posts' in table_name: # si la table est de type posts
            id_column = 'post_id'  # identifiant des posts
        elif 'comments' in table_name:  # si c'est des commentaires
            id_column = 'comment_id'  # identifiant des commentaires
            related_post_column = 'post_id'  # ID du post lié
        else:   # si le type de table n'est pas reconnu
            print(f"Table {table_name} non reconnue. Ignorée.")
            continue  # passer à la suivante

        # charger la table dans un DataFrame
        query = f"SELECT * FROM {table_name}"  # requête SQL
        try:
            df = pd.read_sql_query(query, engine)  # lire les données dans DF
        except Exception as e:   # capture les exceptions
            print(f"Erreur lors de la lecture de la table {table_name}: {e}")
            continue  # passer à la table suivante

        # si la table est vide, passer
        if df.empty:   # si le DF est vide
            print(f"Table {table_name} est vide. Ignorée.")
            continue  # passage table suivante

        # nettoyage pour chaque table (posts ou commentaires)
        if 'posts' in table_name:   # pour posts, nettoie 'title' & 'content'
            # nettoyer 'title' et 'content' d'après options et les concaténer
            df[suffix.lstrip('_')] = df.apply(  # applique fonction nettoyage
                lambda row: ' '.join([  # concatène
                    clean_text(str(row.get('title', '')), args)  # clean titre
                    if pd.notna(row.get('title')) else '',  # si non NaN
                    clean_text(str(row.get('content', '')), args)  # contenu
                    if pd.notna(row.get('content')) else ''  # si non NaN
                ]).strip(),   # supprime les espaces en trop
                axis=1  # applique par ligne
            )
        else:  # pour table de type commentaire
            df[suffix.lstrip('_')] = df['content'].apply( # seulement "content"
                lambda x: clean_text(str(x), args)  # on applique le nettoyage
                if pd.notna(x) else x  # vérifie si le contenu n'est pas vide
            )

        # filtrer les entrées valides (vérifier id non nul)
        if 'comments' in table_name:   # si c'est une table de commentaires
            # conserve les col. id, id du post associé, et la colonne nettoyée
            update_df = df[[id_column, related_post_column,
                            suffix.lstrip('_')]].copy()
        else:   # pour une table de posts
            # conserve les colonnes id et la colonne nettoyée
            update_df = df[[id_column, suffix.lstrip('_')]].copy()
        update_df = update_df[  # filtre lignes avec id non nul et non vide
            update_df[id_column].notnull() &
            (update_df[id_column] != '')
        ]

        # si aucune donnée valide, passer
        if update_df.empty:   # vérifie si le DF est vide après filtrage
            print(f"Table {table_name} n'a pas de données valides à mettre à "
                  f"jour. Ignorée.")
            continue   # passe à la table suivante

        # ajouter la nouvelle colonne si elle n'existe pas
        column_name = suffix.lstrip('_')   # nom de la nouvelle colonne
        try:
            with engine.begin() as conn:   # démarre une transaction
                conn.execute(text(f"""
                    ALTER TABLE {table_name} 
                    ADD COLUMN IF NOT EXISTS "{column_name}" TEXT
                """))  # ajouter colonne si n'existe pas
                conn.execute(text("COMMIT"))  # valide la transaction
                print(f"Colonne '{column_name}' ajoutée à la table "
                      f"'{table_name}'.")   # log succès
        except Exception as e:   # capture les erreurs potentielles
            print(f"Erreur lors de l'ajout de la colonne '{column_name}' "
                  f"à la table '{table_name}': {e}")
            continue   # passe à la table suivante

        # mise à jour des valeurs de la colonne nettoyée
        try:
            with engine.begin() as conn:   # démarre une transaction
                # créer une table temporaire avec les valeurs nettoyées
                temp_table = f"temp_{table_name}"  # nom table temporaire
                update_df.to_sql(
                    temp_table, conn, if_exists='replace', index=False
                )  # sauvegarde données nettoyées dans table temporaire

                # mise à jour des valeurs dans la table principale
                if 'comments' in table_name:   # pour table de commentaires
                    update_query = text(f"""
                        UPDATE {table_name}
                        SET "{column_name}" = temp."{column_name}"
                        FROM {temp_table} AS temp
                        WHERE {table_name}.{id_column} = temp.{id_column}
                        AND {table_name}.{related_post_column} = 
                            temp.{related_post_column}
                    """)   # mise à jour basée sur les ids
                else:   # pour une table de posts
                    update_query = text(f"""
                        UPDATE {table_name}
                        SET "{column_name}" = temp."{column_name}"
                        FROM {temp_table} AS temp
                        WHERE {table_name}.{id_column} = temp.{id_column}
                    """)    # mise à jour basée sur l'id du post
                conn.execute(update_query)  # exécuter la mise à jour

                # supprimer la table temporaire
                conn.execute(text(f"DROP TABLE {temp_table}"))  # suppression
                print(f"Table '{table_name}' mise à jour avec la colonne "
                      f"'{column_name}'.")  # log succès
        except Exception as e:   # capture les erreurs potentielles
            print(f"Erreur lors de la mise à jour de la table "
                  f"'{table_name}': {e}")
            continue   # passe à la table suivante

        print(f"Traitement de la table {table_name} terminé.")

    print("Toutes les tables ont été traitées avec succès.\n\n")


if __name__ == "__main__":

    #sys.argv = [   # pour la phase de développement arguments
        #'2_cleaning.py',
        #'--tables',
            #'posts_table_month_controversial_11_15',
            #'posts_table_month_top_11_15',
            #'posts_table_year_controversial_11_15',
            #'posts_table_year_top_11_15',
            #'comments_table_month_controversial_11_15',
            #'comments_table_month_top_11_15',
            #'comments_table_year_controversial_11_15',
            #'comments_table_year_top_11_15',
            #'posts_table_year_top_today',
            #'comments_table_year_top_today',
        #'--lowercase',  # convertir en minuscules
        #'--remove_links',  # supprimer les liens
        # '--remove_punctuation',  # supprimer la ponctuation
        #'--remove_stopwords',  # supprimer les mots vides
        #'--remove_numbers',  # supprimer les nombres
        # '--correct_abbreviations',  # corriger les abbréviations
        # '--remove_extra_spaces'  # supprimer les espaces multiples
    #]
    main()
