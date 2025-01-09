"""
9_graph.py

Ce script extrait les sentiments des tables PostgreSQL et les visualise au fil
du temps sous forme de graphiques. Les graphiques sont enregistrés dans le
dossier spécifié en argument ou dans "time_graph" par défaut.

Fonctionnalités :
- Connexion à une base de données PostgreSQL via SQLAlchemy
- Extraction des sentiments positifs et négatifs des posts et commentaires
- Agrégation des sentiments par période temporelle (jour ou mois)
- Génération de graphiques de sentiments avec matplotlib
- Enregistrement des graphiques avec des noms de fichiers nettoyés

Exemple d'utilisation :
    python 9_graph.py --output_dir ../sorties/graph_time
"""
import os  # gestion des opérations système
import re  # expressions régulières pour nettoyer les noms de fichiers
import sys  # pour les arguments en ligne de commande
import argparse  # pour le parsing des arguments
import pandas as pd  # manipulation des données
import matplotlib; matplotlib.use('Agg')   # backend non interactif 'Agg'
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text  # connexion et requêtes sql

def get_sqlalchemy_engine():
    """
    Crée et retourne un moteur sqlalchemy pour se connecter à postgresql

    retourne :
        sqlalchemy Engine: moteur sqlalchemy configuré
    """
    # chaîne de connexion postgresql
    connexion_str = (
        "postgresql://avrile:projet@localhost:5432/projet_reddit"
    )
    # crée et retourne le moteur sqlalchemy
    return create_engine(connexion_str)

def extract_and_plot_sentiments(tables, engine, time_category,
                                start_date=None, end_date=None,
                                output_directory='time_graph'):
    """
    Extrait les sentiments de plusieurs tables et les visualise au fil du
    temps. Enregistre les graphiques dans le dossier spécifié

    args:
        tables (list): noms des tables à analyser
        engine (sqlalchemy.engine.Engine): moteur sqlalchemy pour la connexion
        time_category (str): catégorie temporelle pour le titre du graphique
        start_date (str, optional): date de début au format 'yyyy-mm-dd'
        end_date (str, optional): date de fin au format 'yyyy-mm-dd'
        output_directory (str, optional): dossier de sortie pour les graphiques
    """
    # dataframe combiné initial vide
    combined_data = pd.DataFrame()

    # conversion des dates si spécifiées
    if start_date:
        start_date = pd.to_datetime(start_date)
    if end_date:
        end_date = pd.to_datetime(end_date)

    # parcourt chaque table pour extraire les données
    for table_name in tables:
        # détermine la colonne id en fonction du type de table
        if 'posts' in table_name:
            id_column = 'post_id'  # colonne id pour posts
        else:
            id_column = 'comment_id'  # colonne id pour commentaires

        model_column = 'lower_zero-shot'  # colonne du modèle

        try:
            # construit la requête sql
            query = f"""
                SELECT "{id_column}", "{model_column}", "created"
                FROM "{table_name}"
                WHERE "{model_column}" IS NOT NULL
            """
            params = {}
            # ajoute des filtres de date si spécifiés
            if start_date and end_date:
                query += " AND \"created\" BETWEEN :start_date AND :end_date"
                params = {'start_date': start_date, 'end_date': end_date}

            # créé l'objet texte pour la requête
            model_query = text(query)
            # exécute la requête et charger les données
            model_data = pd.read_sql(model_query, engine, params=params)

            # convertit la colonne 'created' en datetime
            model_data['created'] = pd.to_datetime(model_data['created'])

            # détermine le sentiment prédit à partir de la colonne modèle
            model_data['predicted_sentiment'] = model_data[
                model_column
            ].apply(
                lambda x: 'positive' if 'positive' in x.lower()
                else ('negative' if 'negative' in x.lower() else None)
            )

            # concatène les données extraites au dataframe combiné
            combined_data = pd.concat(
                [combined_data, model_data],
                ignore_index=True
            )
        except Exception as e:
            # affiche l'erreur et continuer avec la prochaine table
            print(
                f"erreur lors de la récupération des données pour "
                f"{table_name}: {e}"
            )
            continue

    # supprime les entrées sans sentiment prédit
    combined_data = combined_data.dropna(subset=['predicted_sentiment'])

    # vérifie si des données existent après nettoyage
    if combined_data.empty:
        print(
            f"Aucune donnée pour {time_category} "
            f"dans la plage de dates spécifiée."
        )
        return

    # ajoute une colonne pour l'agrégation temporelle
    if start_date and end_date and (end_date - start_date).days <= 31:
        combined_data['time_period'] = combined_data['created'].dt.to_period('D')
        time_label = 'jour'  # label pour l'axe x
    else:
        combined_data['time_period'] = combined_data['created'].dt.to_period('M')
        time_label = 'mois'  # label pour l'axe x

    # compte les sentiments par période
    sentiment_counts = combined_data.groupby(
        ['time_period', 'predicted_sentiment']
    ).size().unstack(fill_value=0)

    # assure la présence des colonnes 'positive' et 'negative'
    for sentiment in ['positive', 'negative']:
        if sentiment not in sentiment_counts.columns:
            sentiment_counts[sentiment] = 0

    # calcule le total des sentiments par période
    sentiment_counts['total'] = sentiment_counts[
        ['positive', 'negative']
    ].sum(axis=1)

    # configure la figure du graphique
    plt.figure(figsize=(12, 7))
    # trace les sentiments positifs
    plt.plot(
        sentiment_counts.index.to_timestamp(),
        sentiment_counts['positive'],
        label='commentaires et posts positifs',
        color='green'
    )
    # trace les sentiments négatifs
    plt.plot(
        sentiment_counts.index.to_timestamp(),
        sentiment_counts['negative'],
        label='commentaires et posts négatifs',
        color='red'
    )
    # trace le total des sentiments
    plt.plot(
        sentiment_counts.index.to_timestamp(),
        sentiment_counts['total'],
        label='total des commentaires et posts',
        color='blue',
        linestyle='--'
    )
    # ajoute une ligne horizontale à y=0
    plt.axhline(0, color='black', lw=0.5)

    # définit le titre du graphique
    if start_date and end_date:
        plt.title(
            f'Nombre de commentaires et posts positifs, négatifs et total - '
            f'{time_category} ({start_date.date()} - {end_date.date()})'
        )
    else:
        plt.title(
            f'Nombre de commentaires et posts positifs, négatifs et total - '
            f'{time_category} (2023-11-15 - 2024-11-15 )'
        )

    plt.xlabel(time_label)  # label pour l'axe x
    plt.ylabel('nombre de commentaires et posts')  # label pour l'axe y
    plt.legend()  # affiche la légende
    plt.xticks(rotation=45)  # rotation des ticks de l'axe x
    plt.tight_layout()  # ajuste la mise en page pour éviter chevauchements

    # créé le dossier de sortie s'il n'existe pas
    os.makedirs(output_directory, exist_ok=True)

    # construit le nom du fichier en fonction des dates
    if start_date and end_date:
        filename = f"{time_category}_{start_date.date()}_" \
                   f"{end_date.date()}.png"
    else:
        filename = f"{time_category}_2023-11-15_2024-11-15.png"

    # nettoie le nom du fichier en remplaçant les caractères spéciaux
    filename = re.sub(r'[^\w\-_\. ]', '_', filename).replace(' ', '_')

    # construit le chemin complet du fichier
    filepath = os.path.join(output_directory, filename)

    plt.savefig(filepath)  # enregistre le graphique
    # plt.show()  # affiche le graphique - désactiver pour compta linux

def main(output_directory):
    """
    Fonction principale qui configure les options, se connecte à la base de
    données, et lance l'extraction et la visualisation des sentiments

    args:
        output_directory (str): dossier où sauvegarder les graphiques
    """
    # configure les options d'affichage de pandas
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    engine = get_sqlalchemy_engine()
    # connexion à la base de données

    # définit les catégories et tables associées
    categories_with_tables = {
        'année controversée': [
            'posts_table_year_controversial_11_15',
            'comments_table_year_controversial_11_15'
        ],
        'année top': [
            'posts_table_year_top_11_15',
            'comments_table_year_top_11_15'
        ],
        'mois controversé': [
            'posts_table_month_controversial_11_15',
            'comments_table_month_controversial_11_15'
        ],
        'mois top': [
            'posts_table_month_top_11_15',
            'comments_table_month_top_11_15'
        ],
    }

    # parcourt chaque catégorie pour extraire et tracer les sentiments
    for time_category, tables in categories_with_tables.items():
        if 'mois' in time_category:
            # définit la plage de dates pour les données mensuelles
            start_date = '2024-10-15'
            end_date = '2024-11-15'
            extract_and_plot_sentiments(
                tables, engine, time_category,
                start_date, end_date, output_directory
            )
        else:
            # pas de filtre de dates pour les données annuelles
            extract_and_plot_sentiments(
                tables, engine, time_category,
                output_directory=output_directory
            )

    # message final après génération des graphes
    print("\n\nLes graphes ont été générés et enregistrés dans le répertoire "
          "passé en argument\n\n")

if __name__ == "__main__":
    """
    point d'entrée du script pour lancer la génération des graphiques de
    sentiments
    """
    # exemple d'arguments pour tester le script en phase de développement
    #sys.argv = [
    # '9_graph.py',
    # '--output_dir', '../sorties/graph_time'
    #]

    # configure les arguments de la ligne de commande
    parser = argparse.ArgumentParser(
        description='génération de graphiques de sentiments à partir de '
                    'tables postgresql.'
    )
    # créé l'analyseur d'arguments avec une description

    parser.add_argument(
        '--output_dir', type=str, default='time_graph',
        help='dossier où sauvegarder les graphiques.'
    )
    # ajoute l'argument optionnel --output_dir avec valeur par défaut

    args = parser.parse_args()
    # parse les arguments de la ligne de commande

    # exécute le script avec les arguments fournis
    main(args.output_dir)
    # appel de la fonction principale avec le dossier de sortie