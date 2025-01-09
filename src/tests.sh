#!/bin/bash
# =============================================================================
# Script illustrant le fonctionnement de tous les autres scripts utilisés
# pour le projet
# date: 2024-12-14
# description: Ce script lance les tests sur l'ensemble des scripts du
#              projet
# =============================================================================

# ignore les erreurs (continue l'exécution)
set +e

# étape 0 : configuration de la base de données
echo -e "\nEXECUTION : 0_set_up_db.py"
python3 0_set_up_db.py dump_db.dump

# étape 1 : test de la connexion
echo -e "\nEXECUTION : auxiliaire/0_test_conn.py"
python3 auxiliaire/0_test_conn.py

# étape 2 : scraping des données (exécuté deux fois)
echo -e "\nEXECUTION : 1_scrapping.py (1ère exécution)"
python3 1_scrapping.py \
    --folders ../test/mock_data_today/raw/year_top_today \
    --post_filename posts_year_top_today.csv \
    --comment_filename comments_year_top_today.csv \
    --post_table posts_table_year_top_today \
    --comment_table comments_table_year_top_today \
    --time_filter year \
    --sort_by top \
    --subreddits politics \
    --keywords """trump,harris,donald,president,campaign,democrats,
republicans,ballot,swing state,polls,kamala,primary,
vance,walz,election,elections,maga,jd""" \
    --total_posts 15 \
    --comments_per_post 5

    # 2ème exécution pour illuster les erreurs (tables existent déjà)
echo -e "\nEXECUTION : 1_scrapping.py (2ème exécution)"
python3 1_scrapping.py \
    --folders ../test/mock_data_today/raw/year_top_today \
    --post_filename posts_year_top_today.csv \
    --comment_filename comments_year_top_today.csv \
    --post_table posts_table_year_top_today \
    --comment_table comments_table_year_top_today \
    --time_filter year \
    --sort_by top \
    --subreddits politics \
    --keywords """trump,harris,donald,president,campaign,democrats,
republicans,ballot,swing state,polls,kamala,primary,
vance,walz,election,elections,maga,jd""" \
    --total_posts 15 \
    --comments_per_post 5

# étape 3 : nettoyage des données (raw et lower)
echo -e "\nEXECUTION : 2_cleaning.py (raw)"
python3 2_cleaning.py \
    --tables \
    posts_table_year_top_today \
    comments_table_year_top_today

echo -e "\nEXECUTION : 2_cleaning.py (lower)"
python3 2_cleaning.py \
    --tables \
    posts_table_year_top_today \
    comments_table_year_top_today \
    --lowercase

# étape 4 : shuffle des données vers CSV (préparation évaluation manuelle)
echo -e "\nEXECUTION : auxiliaire/1_shuffle_to_csv.py (1ère exécution - posts)"
python3 auxiliaire/1_shuffle_to_csv.py \
    --source_table \
    posts_table_year_top_today \
    --output_csv \
    ../../test/mock_data_today/manual_eval/shuffled_posts_year_top_today.csv

    #2nd appel pour la table de commentaires
echo -e "\nEXECUTION : auxiliaire/1_shuffle_to_csv.py (2ème exécution - commentaires)"
python3 auxiliaire/1_shuffle_to_csv.py \
    --source_table \
    comments_table_year_top_today \
    --output_csv \
    ../../test/mock_data_today/manual_eval/shuffled_comments_year_top_today.csv


# étape 5 : ajout d'évaluations manuelles à la base de données
echo -e "\nEXECUTION : 3_add_manual_eval_bdd.py (1ère exécution)\n"
python3 3_add_manual_eval_bdd.py \
    --tables \
    posts_table_year_top_today \
    comments_table_year_top_today \
    --csv_files \
    ../data/manual_eval/evaluation_manuelle_supplementaire.csv

    # 2nde exécution : rien à rajouter
echo -e "\nEXECUTION : 3_add_manual_eval_bdd.py (2ème exécution - rien à rajouter)\n"
python3 3_add_manual_eval_bdd.py \
    --tables \
    posts_table_year_top_today \
    comments_table_year_top_today \
    --csv_files \
    ../data/manual_eval/evaluation_manuelle_supplementaire.csv

# étape 6 : application des modèles sur toutes les données de test lower
echo -e "\nEXECUTION : 4_apply_model_on_all_data.py (1ère exécution - posts)"
python3 4_apply_model_on_all_data.py \
    --tables posts_table_year_top_today \
    --columns lower \
    --models distilbert zero-shot \
    --primary_key post_id

echo -e "\nEXECUTION : 4_apply_model_on_all_data.py (2ème exécution - commentaires)"
python3 4_apply_model_on_all_data.py \
    --tables comments_table_year_top_today \
    --columns lower \
    --models distilbert zero-shot \
    --primary_key comment_id

# étape 7 : évaluation des performances des modèles (entrées évaluées
# manuellement)

  # run 1 : sur les données tests
echo -e "\nEXECUTION : 5_models_perfs_on_manually_evaluated_entries.py
        (1ère exécution - données tests)"
python3 5_models_perfs_on_manually_evaluated_entries.py \
    --tables \
    posts_table_year_top_today \
    comments_table_year_top_today \
    --output_directory \
    ../test/mock_data_today/sorties/models_perfs_top_today


# à partir d'ici, reprise sur les "vraies" données utilisées pour le projet

  # run 2 : sur les données réelles du projets
echo -e "\nEXECUTION : 5_models_perfs_on_manually_evaluated_entries.py
        (2ème exécution - données du projet)"
python3 5_models_perfs_on_manually_evaluated_entries.py \
    --tables \
    posts_table_month_controversial_11_15 \
    posts_table_month_top_11_15 \
    posts_table_year_controversial_11_15 \
    posts_table_year_top_11_15 \
    comments_table_month_controversial_11_15 \
    comments_table_month_top_11_15 \
    comments_table_year_controversial_11_15 \
    comments_table_year_top_11_15 \
    --output_directory \
    ../test/sorties/models_perfs_project_data

# étape 8: tri des données positives (préparation évaluations manuelle
# supplémentaires avant fine-tunage)

  # run 1 : sur les données de notre projet
echo -e "\nEXECUTION : auxiliaire/2_parse_positive_entries_for_manual_verification.py
        (1ère exécution - données du projet)\n"
python3 auxiliaire/2_parse_positive_entries_for_manual_verification.py \
    --tables \
    posts_table_month_controversial_11_15 \
    posts_table_month_top_11_15 \
    posts_table_year_controversial_11_15 \
    posts_table_year_top_11_15 \
    comments_table_month_controversial_11_15 \
    comments_table_month_top_11_15 \
    comments_table_year_controversial_11_15 \
    comments_table_year_top_11_15 \
    --output  \
    ../../test/manual_eval/positive_entries_posts_et_comments.csv  \
    --extract_type  \
     both

  # run2: sur les données extraites au cours du test
echo -e "\nEXECUTION : auxiliaire/2_parse_positive_entries_for_manual_verification.py
        (1ère exécution - données tests)"
python3 auxiliaire/2_parse_positive_entries_for_manual_verification.py \
    --tables \
    posts_table_year_top_today \
    comments_table_year_top_today \
    --output  \
    ../../test/mock_data_today/manual_eval/positive_entries_posts_et_comments.csv  \
    --extract_type  \
     both

# étape 9 : modèle personnalisé (sur comments et posts)
echo -e "\nEXECUTION : 6_my_fine_model.py (1ère exécution - commentaires)"
python3 6_my_fine_model.py \
    --tables \
    comments_table_month_controversial_11_15 \
    comments_table_month_top_11_15 \
    comments_table_year_controversial_11_15 \
    comments_table_year_top_11_15 \
    --output_dir \
    ../test/fine_tuned/fine_tuned_comments

echo -e "\nEXECUTION : 6_my_fine_model.py (2ème exécution - posts)"
python3 6_my_fine_model.py \
    --tables \
    posts_table_month_controversial_11_15 \
    posts_table_month_top_11_15 \
    posts_table_year_controversial_11_15 \
    posts_table_year_top_11_15 \
    --output_dir \
    ../test/fine_tuned/fine_tuned_posts


# étape 10 : répartition des émotions
echo -e "\nEXECUTION : 7_emotion_repartition.py"
python3 7_emotion_repartition.py \
    --tables \
    posts_table_month_controversial_11_15 \
    posts_table_month_top_11_15 \
    posts_table_year_controversial_11_15 \
    posts_table_year_top_11_15 \
    comments_table_month_controversial_11_15 \
    comments_table_month_top_11_15 \
    comments_table_year_controversial_11_15 \
    comments_table_year_top_11_15 \
    --output ../test/sorties/emotion_repartition.csv

# étape 11 : calcul TF-IDF (lent)
echo -e "\nEXECUTION : 8_tdidf.py"
python3 8_tdidf.py \
    --tables \
    posts_table_month_controversial_11_15 \
    posts_table_month_top_11_15 \
    posts_table_year_controversial_11_15 \
    posts_table_year_top_11_15 \
    comments_table_month_controversial_11_15 \
    comments_table_month_top_11_15 \
    comments_table_year_controversial_11_15 \
    comments_table_year_top_11_15 \
    --analyze_posts \
    --analyze_comments \
    --pos NOUN PROPN ADJ ADV \
    --output_dir ../test/sorties/tdidf_plots

#  étape 12 : génération des graphes
echo -e "\nEXECUTION : 9_graph.py"
python3 9_graph.py \
    --output_dir ../test/sorties/graph_time
