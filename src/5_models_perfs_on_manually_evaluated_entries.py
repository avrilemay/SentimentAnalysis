"""
5_models_perfs_on_manual_evaluated_entries.py

Évalue les performances des modèles d'après les évaluations manuelles.
Compare les évaluations manuelles et les prédictions des modèles
pour évaluer la précision.

Fonctionnalités :
- connexion à PostgreSQL
- récupération des évaluations manuelles et des prédictions des modèles
- calcul de la précision des modèles

Entrée :
- liste des tables à analyser

Sortie :
- précision des modèles présentée dans des tableaux
"""

import pandas as pd  # manipulation de données
import argparse  # analyse des arguments CLI
from sqlalchemy import create_engine, text  # connexion BDD et requêtes
import sys  # gestion des arguments du script
import os  # output CSV (chemin)

def get_sqlalchemy_engine():
    """
    Crée et retourne un moteur SQLAlchemy pour PostgreSQL
    """
    # chaîne de connexion pour la bdd PostgreSQL
    connexion_str = "postgresql://avrile:projet@localhost:5432/projet_reddit"
    return create_engine(connexion_str)  # retourne engine SQLAlchemy


def extract_sentiment(evaluation):
    """
    Extrait le sentiment ('positive', 'negative' ou None) d'une évaluation
    """
    if 'positive' in evaluation.lower():   # si 'positive' présent dans éval
        return 'positive'       # retourne 'positive'
    elif 'negative' in evaluation.lower():    # si 'négative' présent dans éval
        return 'negative'           # retourne 'negative'
    return None     # retourne None si pas de sentiment trouvé


def main():
    """
    Analyse la performance des modèles sur les tables spécifiées
    """
    parser = argparse.ArgumentParser(  # initialise parser pour arg en CLI
        description=(   # description script
            "Analyse la performance des modèles sur les tables "
            "de la base de données."
        )
    )
    parser.add_argument(   # ajoute un arg requis pour spécifier les tables
        '--tables',         # nom argument
        nargs='+',          # accepte une liste de valeurs
        required=True,          # argument obligatoire  et description
        help='Liste des tables à analyser (e.g., posts, comments)'
    )
    parser.add_argument(   # ajoute arg optionnel pour répertoire sortie CSV
        '--output_directory', type=str, default='../sorties/perf_models',
        help='Chemin du répertoire où enregistrer les fichiers CSV de sortie'
    )

    args = parser.parse_args()   # analyse les arg fournis en CLI

    output_dir = args.output_directory   # récupère le répertoire de sortie
    # création du répertoire de sortie s'il n'existe pas:
    os.makedirs(output_dir, exist_ok=True)

    # options d'affichage pandas pour éviter les troncatures
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # crée une connexion à la bdd SQLAlchemy
    engine = get_sqlalchemy_engine()
    # init listes pour stocker: résultats globaux, des posts et des coms
    overall_results = []
    comments_results = []
    posts_results = []

    # boucle sur chaque table spécifiée en argument
    for table_name in args.tables:
        # détermine la colonne d'identifiant selon le type de table
        id_column = 'post_id' if 'posts' in table_name else 'comment_id'

        # récupérer évaluations manuelles depuis la bdd
        try:
            with engine.connect() as connection:
                manual_query = text(f"""
                    SELECT {id_column}, manual_evaluation
                    FROM {table_name}
                    WHERE manual_evaluation IS NOT NULL
                """)
                # exécute requête SQL et charge données dans un DF
                manual_data = pd.read_sql(manual_query, connection)
        except Exception as e:
            print(  # affiche un message d'erreur si la requête échoue
                f"Erreur lors de la récupération des évaluations manuelles dans "
                f"{table_name}: {e}"
            )
            continue   # passe à la prochaine table

        # vérifie si aucune entrée n'a été récupérée
        if manual_data.empty:
            print(f"\nAucune évaluation manuelle dans {table_name}.")
            continue  # saute au prochain itératif

        # applique la fonction extract_sentiment pour extraire les sentiments
        manual_data['manual_sentiment'] = manual_data['manual_evaluation'].apply(
            extract_sentiment
        )
        # supprime la colonne initiale d'évaluation manuelle
        manual_data.drop(columns=['manual_evaluation'], inplace=True)

        # récupérer colonnes de la table
        try:
            with engine.connect() as connection:
                columns_query = text(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                """)
                # exécute la requête pour obtenir les noms de colonnes
                table_columns = connection.execute(columns_query).fetchall()
                # extrait les noms des colonnes dans une liste
                table_columns = [col[0] for col in table_columns]
        except Exception as e:
            print(  # affiche un message d'erreur si la requête échoue
                f"Erreur lors de la récupération des colonnes de la table "
                f"{table_name}: {e}"
            )
            continue   # saute au prochain itératif (table suivante)

        # filtre colonnes correspondant aux modèles avec suffixes spécifiques
        model_columns = [  # on garde _distilbert et _zero-shot
            col for col in table_columns
            if col.endswith('_distilbert') or col.endswith('_zero-shot')
        ]
        # vérifie si aucune colonne de modèle n'est présente
        if not model_columns:
            print(f"Aucune colonne de modèle dans {table_name}.")
            continue   # passe à la prochaine table

        # initialise un dictionnaire pour stocker les prédictions des modèles
        model_predictions = {}
        for model_name in model_columns:  # boucle sur chaque col de modèle
            try:
                with engine.connect() as connection:
                    model_query = text(f"""
                        SELECT {id_column}, "{model_name}"
                        FROM {table_name}
                        WHERE "{model_name}" IS NOT NULL
                    """)
                    # exécute requête et charge les données dans un DF
                    model_data = pd.read_sql(model_query, connection)
            except Exception as e:
                print(  # affiche un message d'erreur si la requête échoue
                    f"Erreur lors de la récupération des prédictions pour "
                    f"{model_name} dans {table_name}: {e}"
                )
                continue   # passe à la prochaine colonne

            # vérifie si aucune donnée n'a été récupérée pour le modèle
            if model_data.empty:
                print(
                    f"Aucune prédiction pour {model_name} dans {table_name}."
                )
                continue   # saute au prochain modèle

            # applique la fonction extract_sentiment sur les prédictions
            model_data['predicted_sentiment'] = model_data[model_name].apply(
                extract_sentiment
            )
            # supprime les lignes où le sentiment prédit est manquant
            model_data = model_data.dropna(subset=['predicted_sentiment'])
            # vérifie si aucune prédiction valide n'est disponible
            if model_data.empty:
                print(
                    f"Aucune prédiction valide pour {model_name} dans "
                    f"{table_name}."
                )
                continue   # passe à colonne-modèle suivante

            # ajoute les prédictions valides au dictionnaire des modèles
            model_predictions[model_name] = model_data[[id_column,
                                                        'predicted_sentiment']]
        # vérifie si aucune prédiction valide n'a été trouvée
        if not model_predictions:
            print(f"Aucune prédiction valide trouvée dans {table_name}.")
            continue   # passe à la prochaine table

        # trouve les IDs communs entre évaluations manuelles et prédictions
        model_ids = [set(df[id_column]) for df in model_predictions.values()]
        common_ids = set(manual_data[id_column]).intersection(*model_ids)
        # conserve uniquement données manuelles correspondant aux IDs communs
        manual_df_common = manual_data[manual_data[id_column].isin(
            common_ids)].copy()

        # boucle sur c/ modèle pour comparer sentiments et calculer précision
        for model_name, model_df in model_predictions.items():
            # fusionne les données manuelles et de modèle sur les IDs communs
            merged_df = manual_df_common.merge(
                model_df, on=id_column, how='inner'
            )
            # vérifie si la fusion a généré un DF vide
            if merged_df.empty:
                print(
                    f"Pas de données après fusion pour {model_name} dans "
                    f"{table_name}."
                )
                continue  # passe au modèle suivant

            # ajoute une colonne pour indiquer si le sentiment correspond
            merged_df['match'] = merged_df['manual_sentiment'] == merged_df[
                'predicted_sentiment']
            # compte le nombre de prédictions correctes
            correct = merged_df['match'].sum()
            # calcule le nombre total de prédictions comparées
            total = len(merged_df)
            # calcule la précision en pourcentage
            accuracy = (correct / total) * 100 if total > 0 else 0

            # crée un dictionnaire contenant les résultats pour ce modèle
            result = {
                'Modèle': model_name,
                'Correct': correct,
                'Total': total,
                'Précision (%)': round(accuracy, 2),
                'Table': table_name
            }
            # ajoute aux résultats globaux
            overall_results.append(result)

            # ajoute les résultats spécifiques aux comments ou posts
            if 'comments' in table_name:
                comments_results.append(result)
            elif 'posts' in table_name:
                posts_results.append(result)

    # vérifie si aucun résultat global n'a été collecté
    if not overall_results:
        print("Aucun résultat collecté.")
        return  # termine l'exécution du script

    # crée un DF à partir des résultats globaux
    results_df = pd.DataFrame(overall_results)

    # agrège résultats globaux pour calculer totaux et précision globale
    global_totals = results_df.groupby('Modèle').agg({
        'Correct': 'sum',
        'Total': 'sum'
    }).reset_index()
    # calcule la précision globale en pourcentage
    global_totals['Précision globale (%)'] = (
        global_totals['Correct'] / global_totals['Total'] * 100
    ).round(2)
    # trie les modèles par précision décroissante
    global_totals.sort_values(by='Précision globale (%)', ascending=False,
                              inplace=True)
    # affiche les performances globales des modèles
    print("\nPerformance globale des modèles:")
    print(global_totals[['Modèle', 'Correct', 'Total', 'Précision globale (%)']])

    # vérifie si des résultats pour les 'comments' existent
    if comments_results:
        # crée un DF pour les résultats des 'comments'
        comments_df = pd.DataFrame(comments_results)
        # agrège résultats des 'comments' pour calculer totaux et précision
        comments_totals = comments_df.groupby('Modèle').agg({
            'Correct': 'sum',
            'Total': 'sum'
        }).reset_index()
        # calcule la précision des 'comments' en pourcentage
        comments_totals['Précision comments (%)'] = (
            comments_totals['Correct'] / comments_totals['Total'] * 100
        ).round(2)
        # trie les modèles par précision décroissante pour les 'comments'
        comments_totals.sort_values(by='Précision comments (%)',
                                    ascending=False, inplace=True)
        # affiche les performances des modèles sur les 'comments'
        print("\nPerformance des modèles sur les 'comments':")
        print(comments_totals[['Modèle', 'Correct', 'Total', 'Précision '
                                                             'comments (%)']])
    else:
        # affiche un message si aucune donnée pour les 'comments'
        print("\nAucune donnée pour les 'comments'.")

    # vérifie si des résultats pour les 'posts' existent
    if posts_results:
        # crée un DF pour les résultats des 'posts'
        posts_df = pd.DataFrame(posts_results)
        # agrège les résultats des 'posts' pour calculer totaux et précision
        posts_totals = posts_df.groupby('Modèle').agg({
            'Correct': 'sum',
            'Total': 'sum'
        }).reset_index()
        # calcule la précision des 'posts' en pourcentage
        posts_totals['Précision posts (%)'] = (
            posts_totals['Correct'] / posts_totals['Total'] * 100
        ).round(2)
        # trie les modèles par précision décroissante pour les 'posts'
        posts_totals.sort_values(by='Précision posts (%)', ascending=False,
                                 inplace=True)
        # affiche les performances des modèles sur les 'posts'
        print("\nPerformance des modèles sur les 'posts':")
        print(posts_totals[['Modèle', 'Correct', 'Total', 'Précision posts ('
                                                          '%)']])
    else:
        # affiche un message si aucune donnée pour les 'posts'
        print("\nAucune donnée pour les 'posts'.")

    # exporter les résultats en CSV
    try:
        # chemins complets des fichiers CSV
        global_csv = os.path.join(output_dir, 'perf_models_global.csv')
        comments_csv = os.path.join(output_dir, 'perf_models_comments.csv')
        posts_csv = os.path.join(output_dir, 'perf_models_posts.csv')

        # export des données   - résultats globaux
        global_totals.to_csv(global_csv, index=False, encoding='utf-8-sig')
        if 'comments_totals' in locals() and not comments_totals.empty:
            comments_totals.to_csv(comments_csv, index=False, encoding='utf-8-sig')
        if 'posts_totals' in locals() and not posts_totals.empty:
            posts_totals.to_csv(posts_csv, index=False, encoding='utf-8-sig')

        print(  # message d'information
            f"\nLes résultats ont été exportés avec succès dans le répertoire "
            f"'{output_dir}'.\n\n"
        )
    except Exception as e:  # récupère les erreurs
        print(f"\nErreur lors de l'exportation des résultats en CSV: {e}")


# vérifie si le script est exécuté directement (et non importé comme module)
if __name__ == "__main__":
    # arguments du script (phase développement)
    #sys.argv = [
        #'5_models_perfs_on_manually_evaluated_entries.py',  # nom du script
        #'--tables',  # argument spécifiant les tables à analyser
        # liste des noms de tables à analyser
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
        #'--output_directory', '../sorties/models_perf'    # répertoire de sortie

    #]
    # exécute la fonction principale pour analyser les performances
    main()
