"""
7_emotion_repartition.py

Représentation statistiques des sentiments dans des tables spécifiques d'une
base de données PostgreSQL.

Ce script se connecte à une base de données PostgreSQL, extrait les données des
tables et colonnes indiquées, calcule les pourcentages de sentiments positifs
et négatifs à partir des prédictions des modèles, puis affiche les résultats de
manière organisée.

Fonctionnalités :
- connexion à PostgreSQL
- extraction des données des tables et colonnes spécifiées
- calcul des pourcentages de sentiments positifs et négatifs
- affichage des résultats triés par pourcentage de positifs décroissant

Entrée :
- liste des tables à analyser via la ligne de commande.

Sortie :
- affichage des pourcentages de sentiments par table et par modèle.
"""

import os   # pour créer le répertoire du fichier CSV si besoin
import pandas as pd  # manipulation des données
import argparse  # analyse des arguments CLI
from sqlalchemy import create_engine, text  # connexion et requêtes SQL
import sys  # gestion des arguments du script


def get_sqlalchemy_engine():
    """
    Crée et retourne un moteur SQLAlchemy pour se connecter à PostgreSQL.

    Returns:
        engine: instance du moteur SQLAlchemy
    """
    # chaîne de connexion PostgreSQL
    connexion_str = "postgresql://avrile:projet@localhost:5432/projet_reddit"
    # retourne moteur SQLAlchemy
    return create_engine(connexion_str)


def main():
    """
    Fonction principale qui exécute l'analyse des sentiments sur les tables
    spécifiées de la base de données
    """
    # crée un analyseur d'arguments pour les options CLI
    parser = argparse.ArgumentParser(
        description='Analyse de sentiment sur les tables de la base de données.'
    )
    # ajoute un argument obligatoire pour spécifier les tables à analyser
    parser.add_argument(
        '--tables', nargs='+', required=True,
        help='Liste des tables à analyser (e.g., posts, comments)'
    )
    # ajoute un argument optionnel pour spécifier le fichier CSV de sortie
    parser.add_argument(
        '--output', type=str, default='resultats_sentiments.csv',
        help='Nom du fichier CSV de sortie (par défaut: resultats_sentiments.csv)'
    )
    # parse les arguments passés via la ligne de commande
    args = parser.parse_args()

    # récupère le répertoire de sortie et en fait un chemin absolu
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        try:  # si le répertoire n'existe pas, essaie de le créer
            os.makedirs(output_dir, exist_ok=True)
            print(f"Répertoire '{output_dir}' créé avec succès.")
        except Exception as e:
            print(
                f"Erreur lors de la création du répertoire '{output_dir}': {e}")
            sys.exit(1)  # arrête le script en cas d'erreur

    # configure pandas pour afficher toutes les lignes
    pd.set_option('display.max_rows', None)
    # configure pandas pour afficher toutes les colonnes
    pd.set_option('display.max_columns', None)
    # configure pandas pour une largeur d'affichage illimitée
    pd.set_option('display.width', None)
    # configure pandas pour ne pas tronquer les colonnes larges
    pd.set_option('display.max_colwidth', None)

    # crée une connexion à la base de données PostgreSQL
    engine = get_sqlalchemy_engine()

    # initialise une liste pour stocker les résultats de toutes les tables
    overall_results = []

    # itère sur chaque table spécifiée dans les arguments
    for table_name in args.tables:
        # configure les noms des col pour les tables contenant "posts"
        if 'posts' in table_name:
            id_column = 'post_id'  # colonne ID pour posts
            model_column = 'lower_zero-shot'  # colonne modèle pour posts
        else:  # configure les noms des colonnes pour les autres tables (coms)
            id_column = 'comment_id'  # colonne ID pour comments
            # colonne modèle pour comments:
            model_column = 'lower_no_url_no_numbers_zero-shot'

        # essaie d'extraire et de traiter les données de la table actuelle
        try:
            # construit requête SQL pour récupérer prédictions non nulles
            model_query = text(f"""
                SELECT {id_column}, "{model_column}"
                FROM {table_name}
                WHERE "{model_column}" IS NOT NULL
            """)
            # exécute la requête SQL et charge les résultats dans un DF
            model_data = pd.read_sql(model_query, engine)
            # applique une classification des sentiments
            model_data['predicted_sentiment'] = model_data[
                model_column
            ].apply(
                # identifie si le sentiment est positif ou négatif
                lambda x: 'positive' if 'positive' in x.lower()
                else ('negative' if 'negative' in x.lower() else None)
            )
        except Exception as e:
            # affichage de l'erreur et passage à la table suivante
            print(f"Erreur lors de la récupération des données prédites: {e}")
            continue

        # calcule le nombre total de prédictions dans la table
        total_predicted = len(model_data)
        # compte le nombre de prédictions positives
        positive_count = model_data['predicted_sentiment'].value_counts().get(
            'positive', 0
        )
        # compte le nombre de prédictions négatives
        negative_count = model_data['predicted_sentiment'].value_counts().get(
            'negative', 0
        )
        # calcule le pourcentage de prédictions positives et arrondi
        positive_percentage = (
            round((positive_count / total_predicted) * 100, 2)
            if total_predicted > 0 else 0
        )
        # calcule le pourcentage de prédictions négatives et arrondi
        negative_percentage = (
            round((negative_count / total_predicted) * 100, 2)
            if total_predicted > 0 else 0
        )

        # prépare les résultats de cette table sous forme de dictionnaire
        result = {
            'Table': table_name,  # nom de la table
            'Modèle': model_column,  # nom du modèle
            'Total prédit': total_predicted,  # nombre total de prédictions
            'Positifs (%)': positive_percentage,  # pourcentage de positifs
            'Négatifs (%)': negative_percentage  # pourcentage de négatifs
        }
        # ajoute les résultats de la table à la liste globale
        overall_results.append(result)

    # vérifie si des résultats ont été collectés
    if not overall_results:
        # alerte si aucun résultat n'a été collecté
        print(
            "Aucun résultat n'a été collecté. Veuillez vérifier les requêtes SQL "
            "et les données."
        )
        return  # fin du script

    # crée un DataFrame pandas avec les résultats globaux
    results_df = pd.DataFrame(overall_results)
    # sélectionne les colonnes pour l'affichage final
    results_df = results_df[
        ['Table', 'Modèle', 'Total prédit', 'Positifs (%)', 'Négatifs (%)']
    ]

    # trie les résultats par pourcentage de positifs décroissants
    results_df = results_df.sort_values(
        by=['Positifs (%)'], ascending=False
    )

    # affiche les résultats triés dans la console
    print(
        "\nPourcentage de sentiments positifs et négatifs par table et par "
        "modèle (triés par pourcentage de positifs décroissant):"
    )
    print(results_df)  # Affichage du DataFrame

    # exportation des résultats en CSV
    try:
        results_df.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"\nLes résultats ont été exportés avec succès dans le fichier "
              f"'{args.output}'.\n\n")
    except Exception as e:
        print(f"\nErreur lors de l'exportation des résultats en CSV: {e}")

if __name__ == "__main__":
    # exemple d'arguments par défaut pour le script (phase développement)
    #sys.argv = [
        #'7_emotion_repartition.py',
        #'--tables',
        #'posts_table_month_controversial_11_15',
        #'posts_table_month_top_11_15',
        #'posts_table_year_controversial_11_15',
        #'posts_table_year_top_11_15',
        #'comments_table_month_controversial_11_15',
        #'comments_table_month_top_11_15',
        #'comments_table_year_controversial_11_15',
        #'comments_table_year_top_11_15',
        # '--output' suivi du nom de fichier souhaité
        #'--output', '../sorties/emotion_repartition.csv'
    #]

    # appelle la fonction principale pour exécuter le script
    main()
