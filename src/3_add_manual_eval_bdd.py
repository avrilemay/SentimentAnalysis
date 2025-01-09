"""
3_add_manual_eval_bdd.py

Ce script met à jour des tables d'une base de données PostgreSQL avec des
évaluations manuelles provenant de fichiers CSV. Pour chaque table spécifiée,
une colonne "manual_evaluation" est ajoutée si elle n'existe pas, puis mise à
jour avec les valeurs fournies.

Fonctionnalités :
- Connexion à PostgreSQL
- Création de la colonne "manual_evaluation" si nécessaire
- Mise à jour des évaluations à partir des CSV
- Journalisation des modifications

Entrées attendues :
- Noms des tables à mettre à jour
- Fichiers CSV contenant les évaluations

Exemple d'utilisation :
    python 3_add_manual_eval_bdd.py --tables posts_table comments_table \
        --csv_files ../../data/eval_1.csv ./evaluations/eval_2.csv
"""

import argparse  # analyse des arguments de CLI
import pandas as pd  # manipulation de données
import os  # gestion des chemins de fichiers
import sys  # gestion des arguments système
from sqlalchemy import create_engine, text  # connexion BDD et requêtes SQL


def get_sqlalchemy_engine():
    """
    Crée un moteur SQLAlchemy pour la base de données PostgreSQL.

    Retourne :
        sqlalchemy.engine.Engine: Moteur SQLAlchemy.
    """
    return create_engine(  # connexion PostgreSQL
        "postgresql://avrile:projet@localhost:5432/projet_reddit"
    )


def process_table(engine, table_name, eval_df_filtered, id_columns):
    """
    Traite une table : ajoute la colonne `manual_evaluation` (si nécessaire),
    met à jour les évaluations, et journalise les modifications.

    Arguments :
        engine (sqlalchemy.engine.Engine): Moteur SQLAlchemy
        table_name (str): Nom de la table à mettre à jour
        eval_df_filtered (pd.DataFrame): DataFrame avec les éval. manuelles
        id_columns (list): Colonnes identifiantes

    Retourne :
        int: Nombre de nouvelles évaluations ajoutées
    """
    try:
        with engine.begin() as conn:  # connexion à la base de données
            # ajouter colonne 'manual_evaluation' si elle n'existe pas
            alter_query = f"""
                ALTER TABLE {table_name} 
                ADD COLUMN IF NOT EXISTS manual_evaluation TEXT
            """  # requête SQL pour modifier la table
            conn.execute(text(alter_query))  # exécution de la cmd
            print(f"Colonne 'manual_evaluation' ajoutée à '{table_name}'.")

            # supprimer doublons dans le DataFrame
            eval_df_filtered = eval_df_filtered.drop_duplicates(subset=id_columns)

            # créer table temporaire pour les évaluations
            temp_table = f"temp_{table_name}"   # nom de la table temp
            eval_df_filtered.to_sql(  # écriture du DF dans la table temporaire
                temp_table, conn, if_exists='replace', index=False)

            # conditions de jointure sur les colonnes identifiantes
            join_conditions = " AND ".join(  # création conditions de jointure
                f"{table_name}.{col} = temp.{col}" for col in id_columns
            )

            # compter les nouvelles mises à jour
            count_query = f"""
                SELECT COUNT(*) 
                FROM {table_name} 
                INNER JOIN {temp_table} temp
                ON {join_conditions}
                WHERE {table_name}.manual_evaluation IS NULL
            """  # requête pour compter les nouvelles mises à jour
            result = conn.execute(text(count_query))   # exec requête
            new_updates = result.scalar()    # récupération du résultat

            # mettre à jour la table avec les nouvelles évaluations
            update_query = text(f"""
                UPDATE {table_name}
                SET manual_evaluation = temp.manual_evaluation
                FROM {temp_table} AS temp
                WHERE {join_conditions}
            """)   # requête de mise à jour
            conn.execute(update_query)      # exécution

            # supprimer la table temporaire
            drop_query = f"DROP TABLE {temp_table}"   # requête sup. table
            conn.execute(text(drop_query))     # exécution de la requête
            print(f"'{table_name}' mise à jour avec {new_updates} nouvelles "
                  f"évaluations.")

            return new_updates  # retourner le nombre de mises à jour

    except Exception as e:
        # gérer les erreurs
        print(f"Erreur lors de la mise à jour de '{table_name}': {e}")
        return 0  # retourner 0 en cas d'échec


def main():
    """
    Fonction principale pour mettre à jour les tables avec des évaluations
    manuelles
    """
    # analyse des arguments de la ligne de commande
    parser = argparse.ArgumentParser(
        description='Mettre à jour les tables avec des évaluations manuelles '
                    'depuis des fichiers CSV.'  # description parser
    )
    parser.add_argument('--tables', nargs='+', required=True,
                        help='Liste des tables à mettre à jour')  # tables
    parser.add_argument('--csv_files', nargs='+', required=True,
                        help='Liste des fichiers CSV contenant les '
                             'évaluations manuelles')   # fichiers CSV
    args = parser.parse_args()  # parser les arguments

    # connexion à la base de données
    engine = get_sqlalchemy_engine()  # obtention engine SQLAlchemy
    total_updates = 0  # initialiser le compteur de mises à jour

    # traiter chaque fichier CSV fourni
    for csv_file in args.csv_files:   # boucler sur les fichiers CSV
        # résoudre le chemin relatif en chemin absolu
        csv_path = os.path.abspath(csv_file)   # obtention chemin absolu

        if not os.path.exists(csv_path):   # si le fichier n'existe pas
            print(f"Le fichier CSV '{csv_file}' n'existe pas à l'emplacement "
                  f"spécifié.")
            continue  # passer au fichier suivant

        try:
            # lire le fichier CSV et normaliser les évaluations
            eval_df = pd.read_csv(csv_path)         # lecture du CSV
            eval_df['manual_evaluation'] = (    # normalisation des évals
                eval_df['manual_evaluation'].str.lower()   # en minuscules
            )
            # filtrer les valeurs valides
            eval_df = eval_df[eval_df['manual_evaluation'].isin(
                ['positive', 'negative']  # valeurs acceptées: pos/nég
            )]   # filtrage des évaluations

        except Exception as e:   # gestion des erreurs
            print(f"Erreur lors de la lecture de '{csv_file}': {e}")
            continue  # passer au fichier suivant

        # traiter chaque table spécifiée
        for table_name in args.tables:  # boucle sur les tables
            print(f"\nMise à jour de la table '{table_name}'...")

            # déterminer les colonnes identifiantes selon la table
            if 'posts' in table_name:  # si table de posts
                id_columns = ['post_id']  # la colonne est post_id
                # vérifier présence de 'post_id' dans le CSV
                if 'post_id' not in eval_df.columns:
                    print(f"CSV '{csv_file}' manque 'post_id' pour "
                          f"'{table_name}'.")    # message d'erreur
                    continue  # passer à la table suivante
                # supprimer les lignes avec 'post_id' manquant
                eval_df_filtered = eval_df.dropna(subset=['post_id'])

            elif 'comments' in table_name:   # si table pour les commentaires
                id_columns = ['post_id', 'comment_id']  # double ID
                # vérifier présence de 'post_id' et 'comment_id' dans le CSV
                if not all(col in eval_df.columns for col in id_columns):
                    print(f"CSV '{csv_file}' manque 'post_id' et/ou "
                          f"'comment_id' pour '{table_name}'.")   # erreur
                    continue  # passer à la table suivante
                # supprimer les lignes avec 'post_id' ou 'comment_id' manquant
                eval_df_filtered = eval_df.dropna(subset=id_columns)

            else:    # table non reconnue
                print(f"Table '{table_name}' non reconnue. Ignorée.")
                continue  # ignorer cette table

            if eval_df_filtered.empty:   # si le DF filtré est vide
                print(f"Aucune donnée valide à mettre à jour pour "
                      f"'{table_name}' dans '{csv_file}'.")   # erreur
                continue  # passer table suivante si le DF filtré est vide

            # traiter la table et obtenir le nombre de mises à jour
            new_updates = process_table(   # appel fonction de traitement
                engine, table_name, eval_df_filtered, id_columns
            )
            total_updates += new_updates  # mettre à jour le total

    # afficher le total des mises à jour
    print(f"Total d'entrées manuelles ajoutées : {total_updates}\n\n")


if __name__ == "__main__":
    """
    Point d'entrée du script pour mettre à jour les tables avec des évaluations
    manuelles provenant de fichiers CSV
    """

    #sys.argv = [   # arguments pour la phase de développement
        #'3_add_manual_eval_bdd.py',   # nom du script (pas important)
        #'--tables',         # argument des tables
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
        #'--csv_files',          # les fichiers CSV avec éval manuelle
            #'../data/manual_eval/evaluation_manuelle_supplementaire.csv',
    #]
    main()  # appelle la fonction principale
