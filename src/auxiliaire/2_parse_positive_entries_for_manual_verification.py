"""
2_parse_positive_entries_for_manual_verification.py

Ce script extrait des données depuis une base PostgreSQL, les filtre et les
sauvegarde dans un fichier CSV. Il traite les posts et les commentaires
en fonction des évaluations manuelles et des prédictions du modèle.

Fonctionnalités :
- Connexion à la base de données PostgreSQL via SQLAlchemy
- Extraction des données des tables passées en arguments
- Création d'un échantillon aléatoire des données positives selon leur type
- Sauvegarde des données dans un fichier CSV

Entrées attendues :
- Liste des noms de tables à extraire
- Chemin du fichier CSV de sortie
- Type d'extraction : 'posts', 'comments' ou 'both'

Exemple :
    python extract_and_save_data.py
"""
import argparse  # récupère les arguments en CLI
import re  # expressions régulières
import sys  # passe des arguments au sein du script
import pandas as pd  # manipulation des dataframes
from sklearn.utils import resample  # échantillonnage aléatoire
from sqlalchemy import create_engine, text  # connexion à la base
import warnings
import os  # pour la création du répertoire si besoin
warnings.filterwarnings("ignore", category=FutureWarning)

def get_sqlalchemy_engine():
    """
    Obtient une instance du moteur sqlalchemy pour postgresql.

    returns:
        engine: instance du moteur sqlalchemy connectée à 'projet_reddit'
    """
    # retourne une instance du moteur sqlalchemy connectée à la base
    return create_engine(
            "postgresql://avrile:projet@localhost:5432/projet_reddit"
        )
    # création du moteur sqlalchemy avec la chaîne de connexion

def extract_label(value):
    """
    Extrait le label à partir d'une chaîne de caractères.

    Args:
        value (str): chaîne contenant le label

    Returns:
        str: label extrait ou valeur nettoyée
    """
    # vérifie si la valeur est une chaîne de caractères
    if isinstance(value, str):
        # extrait le premier élément du tuple avec une expression régulière
        match = re.match(r'\(\s*([^\s,]+)\s*,', value)
        # vérifie si l'expression régulière a trouvé une correspondance
        if match:
            return match.group(1)
            # retourne le premier groupe capturé
        else:
            return value.strip()
            # retourne la valeur nettoyée des espaces
    else:
        return value
        # retourne la valeur telle quelle si ce n'est pas une chaîne

def load_data_from_db(tables, engine):
    """
    Charge et concatène les données des tables spécifiées avec conditions.

    args:
        tables (list): liste des noms de tables à extraire
        engine (Engine): instance du moteur sqlalchemy

    returns:
        dataframe: dataframe concaténé contenant les données filtrées.
    """
    # initialise une liste pour stocker les données de toutes les tables
    overall_data = []
    # boucle sur chaque table spécifiée
    for table in tables:
        try:
            # obtenir les colonnes de la table
            with engine.connect() as connection:
                result = connection.execute(
                    text(f"SELECT * FROM {table} LIMIT 1")
                )
                # récupère les noms des colonnes
                columns = result.keys()
                # fin de l'obtention des colonnes

            # construire les conditions where dynamiques
            conditions = []
            # vérifie si la colonne 'manual_evaluation' existe
            if 'manual_evaluation' in columns:
                conditions.append("manual_evaluation IS NOT NULL")
                # ajoute une condition pour 'manual_evaluation'
            # vérifie si la colonne 'lower_zero-shot' existe
            if 'lower_zero-shot' in columns:
                conditions.append('"lower_zero-shot" IS NOT NULL')
                # ajoute une condition pour 'lower_zero-shot'
            # vérifie si la colonne 'lower_no_url_no_numbers_zero-shot' existe
            if 'lower_no_url_no_numbers_zero-shot' in columns:
                conditions.append(
                    '"lower_no_url_no_numbers_zero-shot" IS NOT NULL'
                )
                # ajoute une condition pour 'lower_no_url_no_numbers_zero-shot'

            # vérifie si des conditions ont été ajoutées
            if not conditions:
                print(
                    f"\n\nAucune colonne pertinente dans la table {table}."
                )
                # affiche un message si aucune colonne pertinente
                continue
                # passe à la table suivante

            # joint les conditions avec un opérateur OR
            where_clause = " OR ".join(conditions)
            # construit la clause where

            # construit la requête sql avec la clause where
            query = text(f"""
                SELECT *
                FROM {table}
                WHERE {where_clause}
            """)
            # création de la requête sql

            # exécute la requête et charge les données dans un dataframe
            data = pd.read_sql(query, engine)
            print(f"Données chargées depuis la table {table}.")
            # affiche un message de chargement réussi

            # extrait les labels des colonnes concernées
            if 'manual_evaluation' in data.columns:
                data['manual_evaluation_label'] = data[
                    'manual_evaluation'
                ].apply(extract_label)
                # applique la fonction extract_label sur 'manual_evaluation'
            if 'lower_zero-shot' in data.columns:
                data['lower_zero-shot_label'] = data[
                    'lower_zero-shot'
                ].apply(extract_label)
                # applique la fonction extract_label sur 'lower_zero-shot'
            if 'lower_no_url_no_numbers_zero-shot' in data.columns:
                data['lower_no_url_no_numbers_zero-shot_label'] = data[
                    'lower_no_url_no_numbers_zero-shot'
                ].apply(extract_label)
                # applique la fonction extract_label sur
                # 'lower_no_url_no_numbers_zero-shot'

            # ajoute la colonne 'label' basée sur 'manual_evaluation_label'
            if 'manual_evaluation_label' in data.columns:
                data['label'] = data['manual_evaluation_label']
                # assigne 'manual_evaluation_label' à 'label'
            else:
                data['label'] = None  # valeurs manquantes
                # assigne None si 'manual_evaluation_label' n'existe pas

            # remplit les valeurs manquantes dans 'label' pour les commentaires
            if ('comment_id' in data.columns and
                    'lower_no_url_no_numbers_zero-shot_label' in data.columns):
                data.loc[
                    data['label'].isna(),
                    'label'
                ] = data.loc[
                    data['label'].isna(),
                    'lower_no_url_no_numbers_zero-shot_label'
                ]
                # remplit 'label' avec
                # 'lower_no_url_no_numbers_zero-shot_label'  si manquant

            # remplit les valeurs manquantes dans 'label' pour les posts
            elif ('post_id' in data.columns and
                  'lower_zero-shot_label' in data.columns):
                data.loc[
                    data['label'].isna(),
                    'label'
                ] = data.loc[
                    data['label'].isna(),
                    'lower_zero-shot_label'
                ]
                # remplit 'label' avec 'lower_zero-shot_label' si manquant

            # convertit les identifiants en chaînes de caractères
            if 'comment_id' in data.columns:
                data['comment_id'] = data['comment_id'].astype(str)
                # convertit 'comment_id' en str
            if 'post_id' in data.columns:
                data['post_id'] = data['post_id'].astype(str)
                # convertit 'post_id' en str

            # réinitialise l'index du dataframe
            data = data.reset_index(drop=True)
            # réinitialise l'index en supprimant l'ancien

            # ajoute les données traitées à la liste globale
            overall_data.append(data)
            print(f"Données de la table {table} ajoutées.")
            # affiche un message d'ajout réussi
        except Exception as e:
            print(
                f"Erreur lors de la récupération des données de {table} : {e}"
            )
            # affiche un message d'erreur en cas d'exception

    # vérifie si des données ont été chargées
    if overall_data:
        # concatène les dataframes sans conflit de colonnes
        concatenated_df = pd.concat(
            overall_data, ignore_index=True, sort=False
        )
        print("Toutes les données ont été concaténées.")
        # affiche un message de concaténation réussie
        return concatenated_df
        # retourne le dataframe concaténé
    else:
        print("Aucune donnée n'a été chargée.")
        # affiche un message si aucune donnée n'a été chargée
        return pd.DataFrame()
        # retourne un dataframe vide

def extract_and_save_data(tables, output_file, extract_type='both'):
    """
    Extrait les données des tables spécifiées et les sauvegarde en csv.

    Args:
        tables (list): liste des noms de tables à traiter
        output_file (str): chemin du fichier csv de sortie
        extract_type (str, optional): type d'extraction
            ('posts', 'comments', 'both'). defaults to 'both'

    Side effects:
        - sauvegarde un fichier csv contenant les données filtrées
        - affiche des messages de progression et d'erreur
    """
    # connexion à la base de données
    engine = get_sqlalchemy_engine()
    print("\nConnexion à la base de données établie.")
    # affiche un message de connexion réussie

    # charge les données depuis la base de données
    df = load_data_from_db(tables, engine)
    # appelle la fonction pour charger les données

    # vérifie si le dataframe est vide
    if df.empty:
        print("Aucune donnée trouvée à traiter.")
        # affiche un message si aucune donnée n'a été trouvée
        return
        # quitte la fonction

    # initialise des dataframes pour les échantillons
    manually_evaluated = pd.DataFrame()
    # dataframe pour les données évaluées manuellement
    comments_positive_sample = pd.DataFrame()
    # dataframe pour l'échantillon de commentaires positifs
    posts_positive_sample = pd.DataFrame()
    # dataframe pour l'échantillon de posts positifs

    # filtre les entrées évaluées manuellement si la colonne existe
    if 'manual_evaluation_label' in df.columns:
        manually_evaluated = df[
            df['manual_evaluation_label'].notna()
        ]
        # filtre les lignes avec une évaluation manuelle
        print("Données manuellement évaluées filtrées.")
        # affiche un message de filtrage réussi

    # filtre les commentaires positifs non évalués manuellement
    if extract_type in ['both', 'comments']:
        if ('comment_id' in df.columns and
                'lower_no_url_no_numbers_zero-shot_label' in df.columns):
            comments_positive = df[
                (df['comment_id'].notna()) &
                (df['manual_evaluation_label'].isna()) &
                (df['lower_no_url_no_numbers_zero-shot_label']
                 .str.contains('positive', case=False, na=False))
            ]
            # filtre les commentaires positifs sans évaluation manuelle
            comments_positive_sample = resample(
                comments_positive,
                replace=False,
                n_samples=int(len(comments_positive) * 0.5) or len(comments_positive),
                random_state=42
            )
            # échantillonne 50% des commentaires positifs
            print("Échantillon de commentaires positifs créé.")
            # affiche un message de création d'échantillon

    # filtre les posts positifs non évalués manuellement
    if extract_type in ['both', 'posts']:
        if ('post_id' in df.columns and
                'lower_zero-shot_label' in df.columns):
            posts_positive = df[
                (df['post_id'].notna()) &
                (df['manual_evaluation_label'].isna()) &
                (df['lower_zero-shot_label']
                 .str.contains('positive', case=False, na=False))
            ]
            # filtre les posts positifs sans évaluation manuelle
            posts_positive_sample = resample(
                posts_positive,
                replace=False,
                n_samples=int(len(posts_positive) * 0.5) or len(posts_positive),
                random_state=42
            )
            # échantillonne 50% des posts positifs
            print("Échantillon de posts positifs créé.")
            # affiche un message de création d'échantillon

    # concatène les ensembles filtrés
    final_df = pd.concat(
        [manually_evaluated,
         comments_positive_sample,
         posts_positive_sample],
        ignore_index=True
    )
    # concatène les dataframes filtrés
    print("Données filtrées concaténées.")
    # affiche un message de concaténation réussie

    # définit les colonnes à supprimer
    columns_to_drop = [
        "manual_evaluation_label",
        "lower_zero-shot_label",
        "lower_no_url_no_numbers_zero-shot_label"
    ]
    # liste des colonnes inutiles à supprimer
    final_df = final_df.drop(
        columns=[col for col in columns_to_drop
                 if col in final_df.columns],
        errors='ignore'
    )
    # supprime les colonnes inutiles si elles existent
    print("Colonnes inutiles supprimées.")
    # affiche un message de suppression réussie

    # sauvegarde le dataframe final en csv
    final_df.to_csv(
        output_file, index=False, encoding='utf-8'
    )
    # enregistre le dataframe final dans le fichier csv spécifié
    print(f"Données sauvegardées dans {output_file}\n\n")
    # affiche un message de sauvegarde réussie

if __name__ == "__main__":
    """
    Point d'entrée du script pour extraire et sauvegarder les données depuis la 
    base de données vers un fichier CSV en utilisant des arguments CLI.
    """

    # définit les arguments en ligne de commande
    parser = argparse.ArgumentParser(
        description="Extrait et filtre des données depuis une base PostgreSQL "
                    "et les exporte vers un fichier CSV."
    )
    parser.add_argument(
        '--tables', type=str, nargs='+', required=True,
        help='Liste des noms de tables à extraire. Exemple: --tables table1 '
             'table2 table3'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Chemin et nom du fichier CSV de sortie. Exemple: --output data.csv'
    )
    parser.add_argument(
        '--extract_type', type=str, choices=['posts', 'comments',
                                             'both'], default='both',
        help="Type d'extraction : 'posts', 'comments' ou 'both'. "
             "Par défaut: 'both'."
    )

    #sys.argv = [   # passage des arguments au sein du script pour dév.
            #'2_parse_positive_entries_for_manual_verification.py',
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
            #'--output', '../../test/positive_entries_posts_et_comments.csv',
            #'--extract_type', 'both'
        #]

    # analyser les arguments
    args = parser.parse_args()

    # obtenir le chemin absolu du script
    script_path = os.path.abspath(__file__)

    # obtenir le répertoire contenant le script (sans le nom du script)
    script_dir = os.path.dirname(script_path)

    # vérifie si le chemin de sortie est absolu ou relatif
    output_file = args.output
    if not os.path.isabs(output_file):
        # si relatif, le rendre relatif au répertoire du script
        output_file = os.path.join(script_dir, output_file)
    # sinon, conserve le chemin absolu tel quel

    # récupère uniquement le répertoire de sortie
    output_dir = os.path.dirname(output_file)

    # créé le répertoire s'il n'existe pas
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Répertoire créé : {output_dir}")

    # appelle la fonction principale avec les arguments fournis
    extract_and_save_data(args.tables, output_file, args.extract_type)


