"""
1_shuffle_to_csv.py

Le script se connecte à une base de données PostgreSQL, extrait les données
d'une table spécifiée, mélange aléatoirement les lignes et exporte le
résultat dans un fichier CSV à un emplacement et avec un nom donnés.

Fonctionnalités :
- connexion à une base de données PostgreSQL via SQLAlchemy
- extraction des données d'une table source
- mélange aléatoire des données
- export des données mélangées dans un fichier CSV
- spécification du chemin et du nom du fichier de sortie
- gestion des erreurs et affichage de messages de confirmation

Entrées attendues :
- nom de la table source (source_table)
- chemin et nom du fichier CSV de sortie (output_csv)

Exemple :
    python shuffle_db_to_csv.py --source_table posts
                              --output_csv output/shuffled_posts.csv
"""

import argparse  # parsing des arguments en ligne de commande
import sys  # gestion des arguments système
import os  # gestion des fichiers et répertoires
from sqlalchemy import create_engine, text  # connexion et exécution SQL
import pandas as pd  # manipulation des dataframes


def get_sqlalchemy_engine():
    """
    Obtient une instance du moteur sqlalchemy avec l'URL de connexion intégrée.

    Returns:
        engine: instance du moteur sqlalchemy connectée à la base.
    """
    database_url = (
        'postgresql://avrile:projet@localhost:5432/projet_reddit'
    )
    # retourne une instance du moteur sqlalchemy connectée à la base
    return create_engine(database_url)


def shuffle_table_to_csv(engine, source_table, output_csv):
    """
    Extrait les données d'une table, les mélange, et les exporte vers un CSV.

    Args:
        engine (Engine): moteur SQLAlchemy connecté à la base de données
        source_table (str): nom de la table source
        output_csv (str): chemin complet du fichier CSV de sortie

    Side effects:
        - crée le répertoire de sortie si inexistant
        - sauvegarde un nouveau fichier CSV mélangé
        - affiche des messages de confirmation ou d'erreur
    """
    try:
        # vérifier si la table source existe
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = :table)"
                ),
                {"table": source_table}
            )
            exists = result.scalar()
            if not exists:
                print(
                    f"Erreur : la table source '{source_table}' n'existe pas."
                )
                return

        # extraire les données de la table source dans un dataframe
        df = pd.read_sql_table(source_table, con=engine)
        print(f"\n\nTable '{source_table}' chargée depuis la base de données.")

        # mélanger les lignes du dataframe
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        print("Lignes mélangées.")

        # obtenir le chemin absolu du script
        script_path = os.path.abspath(__file__)

        # obtenir le répertoire contenant le script
        script_dir = os.path.dirname(script_path)

        # vérifier si le chemin de sortie est absolu ou relatif
        if not os.path.isabs(output_csv):
            # si relatif, le rendre relatif au répertoire du script
            output_path = os.path.join(script_dir, output_csv)
        else:
            # si absolu, le conserver tel quel
            output_path = output_csv

        # récupérer uniquement le répertoire de sortie
        output_dir = os.path.dirname(output_path)


        # créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        # message d'info
        print(f"Répertoire créé : {output_dir}")

        # sauvegarder le dataframe mélangé dans un nouveau CSV
        df_shuffled.to_csv(output_path, index=False)
        print(f"Fichier CSV mélangé enregistré : {output_path}\n\n")

    except Exception as e:
        print(f"Une erreur est survenue : {e}")


if __name__ == "__main__":
    """
    Point d'entrée du script pour mélanger les données d'une table DB et
    les exporter vers un fichier CSV
    """

    # définir les arguments en ligne de commande
    parser = argparse.ArgumentParser(
        description="Mélange les données d'une table et les exporte vers un CSV."
    )
    parser.add_argument(
        '--source_table', type=str, required=True,
        help='Nom de la table source dans la base de données.'
    )
    parser.add_argument(
        '--output_csv', type=str, required=True,
        help='Chemin complet du fichier CSV de sortie.'
    )

    #sys.argv = [
            #'1_shuffle_to_csv.py',
            #'--source_table', 'posts_table_year_top_today',
            #'--output_csv', '../../test/data/shuffled_posts_year_top_today.csv'
        #]

    args = parser.parse_args()

    # obtenir le moteur SQLAlchemy
    engine = get_sqlalchemy_engine()

    # appeler la fonction de mélange et exportation
    shuffle_table_to_csv(engine, args.source_table, args.output_csv)