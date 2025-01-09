"""
4_apply_model_on_all_data.py

Analyse de sentiments sur les colonnes spécifiées des tables PostgreSQL.

Ce script applique des modèles de classification de sentiments sur les
colonnes désignées des tables d'une base de données PostgreSQL.
Il peut utiliser les modèles 'distilbert' et/ou 'zero-shot' pour prédire
les sentiments des textes présents dans les colonnes.

Fonctionnalités :
- Connexion à une base de données PostgreSQL
- Chargement des données des tables et colonnes spécifiées
- Application des modèles de classification de sentiments
- Mise à jour des tables avec les prédictions de sentiments
- Options pour filtrer les entrées manuellement évaluées et
  écraser les valeurs existantes

Entrée :
- Liste des tables à analyser
- Liste des colonnes à analyser dans chaque table
- Liste des modèles à utiliser ('distilbert', 'zero-shot')
- Options supplémentaires via la ligne de commande

Sortie :
- Mise à jour des tables PostgreSQL avec les prédictions de sentiments
"""


import argparse  # analyse des arguments CLI
import pandas as pd  # manipulation de données
from sqlalchemy import create_engine, text  # connexion et requêtes SQL
from transformers import (pipeline, AutoTokenizer,
                          AutoModelForSequenceClassification)   # modèles NLP
import torch  # calculs tensoriels
import sys  # gestion des arguments du script
from tqdm import tqdm  # barre de progression


def get_sqlalchemy_engine():
    """
    Crée et retourne un moteur SQLAlchemy pour se connecter à PostgreSQL.

    Retourne :
        engine : Moteur SQLAlchemy pour PostgreSQL.
    """
    # chaîne de connexion PostgreSQL pour la bdd
    connexion_str = "postgresql://avrile:projet@localhost:5432/projet_reddit"
    # création du de l'engine SQLAlchemy
    return create_engine(connexion_str)


def main():
    """
    Fonction principale pour configurer et exécuter l'analyse de sentiments
    sur les tables de la base de données
    """
    # initialisation du parseur d'arguments pour les options CLI
    parser = argparse.ArgumentParser(
        description='Analyse de sentiment sur les tables de la base de données.')
    # ajout de l'argument '--tables' pour spécifier les tables à analyser
    parser.add_argument('--tables', nargs='+', required=True,
            help='Liste des tables à analyser (e.g., posts, comments)')
    # ajout de l'argument '--columns' pour spécifier les colonnes à analyser
    parser.add_argument('--columns', nargs='+', required=True,
            help='Liste des colonnes à analyser dans chaque table')
    # ajout de l'argument '--models' pour spécifier les modèles à utiliser
    parser.add_argument('--models', nargs='+',
            choices=['distilbert', 'zero-shot'], required=True,
            help="Choisir un ou plusieurs modèles: distilbert, zero-shot")
    # option '--manual_only' (analyser que les entrées manuellement évaluées)
    parser.add_argument('--manual_only', action='store_true',
            help="Analyser uniquement les entrées évaluées manuellement")
    # option '--overwrite' pour écraser les valeurs existantes
    parser.add_argument('--overwrite', action='store_true',
            help="Écraser les valeurs existantes dans les colonnes de sortie")
    # spécifie la colonne de clé primaire dans les tables
    parser.add_argument('--primary_key', required=True,
            help='Nom de la colonne clé primaire dans les tables')
    # analyse les arguments fournis par l'utilisateur
    args = parser.parse_args()

    # crée une connexion à la base de données
    engine = get_sqlalchemy_engine()  # obtention de l'engine

    # configure l'appareil (GPU ou CPU) pour l'exécution des modèles
    device = torch.device('cuda') if torch.cuda.is_available() else (
                                                torch.device('cpu'))   # device

    # initialise un dictionnaire pour stocker les modèles
    models = {}

    # configure et charge le modèle DistilBERT si spécifié
    if 'distilbert' in args.models:
        # charge le tokenizer pour le modèle DistilBERT
        tokenizer = AutoTokenizer.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english')
        # charge le modèle DistilBERT pour la classification
        model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english').to(device)
        # ajoute une fonction de classification utilisant DistilBERT au dico
        models['distilbert'] = lambda text: classify_text_distilbert_long(
            text, tokenizer, model, device)

    # configure et charge le pipeline Zero-Shot si spécifié
    if 'zero-shot' in args.models:
        # initialise le pipeline pour la classification Zero-Shot
        classifier = pipeline("zero-shot-classification",
                              model="facebook/bart-large-mnli",
                              device=0 if torch.cuda.is_available() else -1)
        # définit les labels pour la classification de sentiment
        sentiment_labels = ["Positive", "Negative"]
        # ajoute une fonction de classification utilisant Zero-Shot au dico
        models['zero-shot'] = lambda text: classify_text_zero_shot(
            text, classifier, sentiment_labels, max_length=512)

    # boucle sur chaque table spécifiée pour effectuer l'analyse
    for table_name in args.tables:
        # boucle sur chaque colonne spécifiée pour appliquer les modèles
        for column_name in args.columns:
            # affiche le statut du traitement pour chaque table et colonne
            print(f"\n\nTraitement de la table {table_name}, colonne "
                  f"{column_name}...")

            # génère requête SQL pour sélect toutes les données de la table
            query = f"SELECT * FROM {table_name}"
            try:
                # exécute la requête et charge les données dans un DF
                df = pd.read_sql_query(query, engine)
            except Exception as e:
                # affiche message d'erreur en cas de problème avec la requête
                print(f"Erreur lors de la lecture de la table {table_name}: {e}")
                continue  # passage à la table suivante

            # vérifie si le DF est vide
            if df.empty:
                print(f"Table {table_name} est vide. Ignorée.")
                continue # passage à la table suivante

                # filtre uniquement entrées évaluées manuellement si option
            if args.manual_only:
                if 'manual_evaluation' in df.columns:
                    # conserve uniquement lignes avec évaluations manuelles
                    df = df[df['manual_evaluation'].notna()]
                    if df.empty:
                        # ignore les tables sans entrées évaluées
                        print(f"Aucune entrée évaluée manuellement dans la "
                              f"table '{table_name}'. Ignorée.")
                        continue  # passe à la table suivante
                    else:
                        # affiche le nombre d'entrées évaluées trouvées
                        print(f"Entrées évaluées manuellement trouvées : "
                              f"{len(df)} entrées.")
                else:
                    # ignorer si colonne 'manual_evaluation' absente
                    print(f"La colonne 'manual_evaluation' n'existe pas dans "
                          f"la table {table_name}. Ignorée.")
                    continue

            # boucle sur c/ modèle pour appliquer prédictions sur la colonne
            for model_name, classify_function in models.items():
                # définit le nom de la colonne de sortie pour les prédictions
                output_column = f"{column_name}_{model_name}"

                # ajoute une col de sortie dans la table si elle n'existe pas
                with engine.begin() as conn:
                    conn.execute(text(
                        f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS "
                        f"\"{output_column}\" TEXT"))
                    print(f"Colonne '{output_column}' ajoutée à la table '"
                          f"{table_name}'.")

                # recharge les données pour inclure les nouvelles colonnes
                df = pd.read_sql_query(query, engine)  # rechargement

                # vérifie si la colonne de sortie est présente dans les données
                if output_column not in df.columns:
                    # afficher une erreur si la colonne n'est pas ajoutée
                    print(f"Erreur: la colonne '{output_column}' n'a pas été "
                          f"ajoutée correctement à la table '{table_name}'.")
                    continue  # passe au modèle suivant

                # applique le filtre 'manual_only' après rechargement si choisi
                if args.manual_only:
                    if 'manual_evaluation' in df.columns:
                        # conserve uniquement les entrées évaluées manuellement
                        df = df[df['manual_evaluation'].notna()]
                        if df.empty:   # si le DF est vide
                            # ignore si pas d'entrée évaluée après rechargement
                            print(f"Aucune entrée évaluée manuellement dans "
                                  f"la table '{table_name}' après "
                                  f"rechargement. Ignorée.")
                            continue    # passe au modèle suivant
                        else:
                            # affiche nb d'entrées trouvées après rechargement
                            print(f"Entrées évaluées manuellement trouvées "
                                  f"après rechargement : {len(df)} entrées.")
                    else:
                        # ignorer si colonne absente après rechargement
                        print(f"La colonne 'manual_evaluation' n'existe pas "
                        f"dans la table {table_name} après rechargement. "
                              f"Ignorée.")
                        continue   # passe au modèle suivant

                # détermine les entrées à mettre à jour
                df_to_update = df   # initialise le DF à mettre à jour
                if not args.overwrite:
                    # filtre les entrées déjà traitées si 'overwrite' est False
                    df_to_update = df_to_update[df_to_update[output_column].isna()]

                # vérifie si toutes les entrées sont déjà traitées
                if df_to_update.empty:
                    # ignorer si toutes les entrées sont déjà traitées
                    print(f"Toutes les entrées ont déjà été traitées pour la "
                          f"colonne '{output_column}'.")
                    continue   # passe au modèle suivant

                # applique le modèle sur chaque ligne à mettre à jour
                for index in tqdm(df_to_update.index, desc=f"Analyse de la "
                    f"colonne '{column_name}' avec le modèle '{model_name}'"):
                    # récupère la valeur du texte à analyser
                    text_value = str(df_to_update.at[index, column_name])
                    # vérifie si le texte est valide (non vide)
                    if pd.notna(text_value) and text_value.strip():
                        # applique la fonction de classification au texte
                        sentiment, score = classify_function(text_value)
                        # met à jour la prédiction dans le DF
                        df_to_update.at[index, output_column] = sentiment
                    else:
                        # affiche un message si le texte est vide ou invalide
                        print(f"Entrée vide ou invalide ignorée à l'index "
                              f"{index}.")

                # met à jour la table PostgreSQL avec les nouvelles prédictions
                try:
                    with engine.begin() as conn:
                        # crée une table temporaire pour les mises à jour
                        temp_table = f"temp_{table_name}"
                        # exporter les data à màj de temps vers df
                        df_to_update[[args.primary_key, output_column]].to_sql(
                            temp_table, conn, if_exists='replace', index=False)
                        # génère la requête SQL pour mettre à jour la table
                        update_query = text(f"""
                            UPDATE {table_name}
                            SET "{output_column}" = temp."{output_column}"
                            FROM {temp_table} AS temp
                            WHERE {table_name}."{args.primary_key}" = 
                                temp."{args.primary_key}"
                        """)
                        # exécute la requête de mise à jour (update)
                        conn.execute(update_query)
                        # exécute la requête de mise à jour (drop table)
                        conn.execute(text(f"DROP TABLE {temp_table}"))
                        print(f"Table '{table_name}' mise à jour avec la "
                              f"colonne '{output_column}'.")
                except Exception as e:
                    # affiche un message en cas d'erreur lors de la màj
                    print(f"Erreur lors de la mise à jour de la table '"
                          f"{table_name}': {e}")
                    continue  # passe au modèle suivant

                # affiche un message lorsque toutes les tables ont été traitées
                print("Toutes les tables ont été traitées avec succès.\n\n")


def classify_text_distilbert_long(text, tokenizer, model, device):
    """
    Classifie un texte long en utilisant le modèle DistilBERT en le découpant
    en segments si nécessaire.

    Arguments :
        text (str) : texte à classifier
        tokenizer : tokenizer du modèle DistilBERT
        model : modèle DistilBERT pour la classification
        device : device pour le calcul (CPU ou GPU)

    Retourne :
        tuple : (sentiment, score) où 'sentiment' est 'positive' ou 'negative',
                et 'score' est la différence des probabilités
    """
    max_length = 512  # définit la longueur maximale des tokens
    tokens = tokenizer.encode(text, truncation=False)  # tokenise le texte
    # divise les tokens en segments de taille maximale
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens),
                                                      max_length)]

    scores = []  # initialise une liste pour les scores
    for chunk in chunks:
        if len(chunk) > max_length:
            chunk = chunk[:max_length]  # tronque les segments trop longs
        inputs = torch.tensor([chunk]).to(device)  # convertit en tenseur
        with torch.no_grad():
            outputs = model(inputs)  # génère les prédictions
            # applique la fonction softmax pour obtenir les probabilités
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # extrait le score positif
            positive_score = probabilities[0][1].item()
            # extrait le score négatif
            negative_score = probabilities[0][0].item()
            # calcule la différence
            scores.append(positive_score - negative_score)

    avg_score = sum(scores) / len(scores)  # moyenne des scores
    # détermine le sentiment
    final_sentiment = 'positive' if avg_score > 0 else 'negative'
    # retourne le sentiment et le score
    return final_sentiment, avg_score


def classify_text_zero_shot(text, classifier, sentiment_labels, max_length=512):
    """
    Classifie un texte en utilisant le modèle Zero-Shot en le découpant
    en segments si nécessaire.

    Arguments :
        text (str) : texte à classifier
        classifier : pipeline Zero-Shot
        sentiment_labels (list) : labels de sentiments
        max_length (int) : longueur maximale des segments

    Retourne :
        tuple : (sentiment, score) où 'sentiment' est 'positive' ou 'negative',
                et 'score' est la différence des probabilités
    """
    # nettoie le texte en supprimant les espaces inutiles
    text = text.strip()
    if not text:
        return None, None  # retourne None si le texte est vide

    # divise le texte en segments pour éviter de dépasser la longueur maximale
    segments = [text[i:i + max_length] for i in range(0, len(text), max_length)]

    scores = []  # initialise une liste pour stocker les scores
    for segment in segments:
        # classification du segment
        result = classifier(segment, candidate_labels=sentiment_labels)
        # récupérer les scores
        positive_score = result['scores'][result['labels'].index('Positive')]
        negative_score = result['scores'][result['labels'].index('Negative')]
        scores.append(positive_score - negative_score)  # calcule la différence

    # moyenne des scores des segments
    avg_score = sum(scores) / len(scores)
    sentiment = 'positive' if avg_score > 0 else 'negative'  # sentiment final
    # retourne le sentiment et le score moyen
    return sentiment, avg_score


# point d'entrée pour exécuter le script
if __name__ == "__main__":
    # arguments par défaut lors de la phase de développement
    #sys.argv = [
        #'4_apply_model_on_all_data.py',
        #'--tables',  # les tables à analyser [coms OU posts, pas les 2]
        #'comments_table_month_controversial_11_15',
        #'comments_table_month_top_11_15',
        #'comments_table_year_controversial_11_15',
        #'comments_table_year_top_11_15',
        #'posts_table_year_top_today',
        #'--columns',  # spécifie les colonnes contenant les textes à analyser
        #'lower',
        #'--models',  # les modèles à utiliser [distilbert / zero-shot / les 2]
        #'distilbert', 'zero-shot',
        #'--overwrite',
        #'--primary_key',  # les clés primaires: [comment_id] OU [post_id]
        #'post_id',
    #]
    # exécution de la fonction principale
    main()
