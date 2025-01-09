"""
6_my_fine_model.py

Ce script entraîne un modèle de classification de séquences avec DistilBERT
en utilisant des données de Reddit stockées dans une base PostgreSQL. Il
charge et prétraite les données, entraîne le modèle, évalue ses performances
et sauvegarde le modèle pour une utilisation ultérieure.

Fonctionnalités :
- connexion à PostgreSQL
- chargement et prétraitement des données
- entraînement avec DistilBERT
- évaluation sur validation et test
- sauvegarde du modèle

Entrée :
- liste des tables à traiter

Sortie :
- modèle entraîné
- métriques d'évaluation
"""

import os   # chemin fichier (output)
import argparse  # gestion des arguments CLI
import pandas as pd  # manipulation de données
import torch  # utilisation de modèles PyTorch
from transformers import (  # modules transformers pour NLP et entraînements
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)  # Modèles NLP
from sklearn.metrics import classification_report  # évaluation des modèles
from sklearn.model_selection import train_test_split  # séparation des données
from torch.utils.data import Dataset  # création de datasets personnalisés
from sqlalchemy import create_engine, text  # connexion à la base de données
import sys  # accès aux arguments système


# détecte l'appareil disponible : GPU (MPS/CUDA) ou CPU
device = torch.device(
    "mps" if torch.backends.mps.is_available() else   # mac m1/m2
    "cuda" if torch.cuda.is_available() else    # gpu nvidia
    "cpu"           # sinon
)


# classe personnalisée pour représenter données Reddit sous forme de Dataset
class RedditDataset(Dataset):
    """
    Dataset personnalisé pour les données Reddit

    Args:
        encodings (dict): encodages des textes
        labels (list): labels associés aux textes
    """
    def __init__(self, encodings, labels):
        # initialise les encodages des textes
        self.encodings = encodings
        # initialise les labels associés
        self.labels = labels

    def __getitem__(self, idx):
        # récupère un élément à l'index donné
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        # ajoute les labels correspondants
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        # retourne la taille totale du dataset
        return len(self.labels)


def get_sqlalchemy_engine():
    """
    Crée et retourne un moteur SQLAlchemy pour se connecter à PostgreSQL

    Returns:
        sqlalchemy engine: Moteur SQLAlchemy pour PostgreSQL
    """
    # retourne engine SQLAlchemy avec chaîne de connexion PostgreSQL
    return create_engine("postgresql://avrile:projet@localhost:5432/projet_reddit")


def load_data_from_db(tables, engine):
    """
    Charge les données depuis les tables spécifiées dans la base de données

    Args:
        tables (list): liste des noms de tables à charger
        engine (sqlalchemy engine): moteur SQLAlchemy pour la connexion

    Returns:
        pd.DataFrame: un DataFrame contenant les données chargées.
    """
    # initialise une liste pour stocker les données de chaque table
    overall_data = []
    # boucle sur chaque table spécifiée
    for table in tables:
        try:
            # requête SQL pour charger données avec évaluations manuelles
            query = text(f"""
                SELECT *
                FROM {table}
                WHERE manual_evaluation IS NOT NULL
            """)
            # exécute la requête et charge les résultats dans un DF
            data = pd.read_sql(query, engine)
            # renomme les colonnes pour éviter les conflits
            data = data.rename(
                columns=lambda col: f"{table}_{col}"
                if col not in ['no_punct', 'manual_evaluation']
                else col
            )
            # renomme 'manual_evaluation' en 'label' pour cohérence des labels
            data = data.rename(columns={'manual_evaluation': 'label'})
            # réinitialise l'index du DF
            data = data.reset_index(drop=True)
            # ajoute les données de la table à la liste
            overall_data.append(data)
        except Exception as e:
            # affiche un message d'erreur si le chargement échoue
            print(f"Erreur lors de la récupération de {table}: {e}")

    # combine les données de toutes les tables en un seul DF
    if overall_data:
        return pd.concat(overall_data, ignore_index=True, sort=False)
    else:
        # retourne un DF vide si aucune donnée n'est chargée
        return pd.DataFrame()


def main():
    """
    Fonction principale qui exécute le processus d'entraînement et d'évaluation.
    """
    # initialise un parseur d'arguments pour la ligne de commande
    parser = argparse.ArgumentParser(
        description='Analyse de sentiment sur les tables de la base de données.'
    )

    # ajoute un argument pour spécifier les tables à analyser
    parser.add_argument(
        '--tables', nargs='+', required=True,
        help='Liste des tables à analyser (e.g., posts, comments)'
    )

    parser.add_argument(   # ajoute un arg pour le chemin du dossier sortie
        '--output_dir', type=str, required=True,
        help='Chemin du dossier où enregistrer les résultats (ex: ./test/).'
    )

    # analyse les arguments fournis par l'utilisateur
    args = parser.parse_args()

    # crée une connexion à la base de données avec SQLAlchemy
    engine = get_sqlalchemy_engine()

    # charge les données des tables spécifiées par l'utilisateur
    df = load_data_from_db(args.tables, engine)
    # vérifie si des données ont été chargées
    if df.empty:
        print("Aucune donnée chargée. Vérifiez les tables spécifiées et les "
              "conditions de filtrage.")
        return

    # convertit les labels en valeurs numériques : 1 'positive', 0 'negative'
    df['label'] = df['label'].apply(
        lambda x: 1 if 'positive' in x.lower() else 0
    )

    # vérifie la présence et la validité de la colonne contenant les textes
    if 'no_punct' not in df.columns:
        print("La colonne 'no_punct' est absente des données. Assurez-vous que "
              "les tables contiennent cette colonne.")
        return
    # filtre les lignes avec des valeurs non nulles dans la colonne 'no_punct'
    df = df[df['no_punct'].notna()]

    # affiche la distribution des classes (pos et nég) avant équilibrage
    print("\n\nDistribution des classes avant équilibrage :")
    print(df['label'].value_counts())

    # sépare les classes majoritaires (négatives) et minoritaires (positives)
    df_negative = df[df['label'] == 0]  # données avec label négatif
    df_positive = df[df['label'] == 1]   # données avec label positif

    # récupère le nb d'échantillons à équilibrer (min entre les deux classes)
    num_samples = min(len(df_negative), len(df_positive))

    # échantillonne un nombre égal de positifs et négatifs
        # entrées évaluées négativement
    df_negative_balanced = df_negative.sample(n=num_samples, random_state=42)
        # entrées évaluées positivement
    df_positive_balanced = df_positive.sample(n=num_samples, random_state=42)

    # combine les deux classes équilibrées
    df_balanced = pd.concat([df_negative_balanced, df_positive_balanced]).sample(
        frac=1, random_state=42  # mélange les données aléatoirement
    ).reset_index(drop=True)

    # divise les données équilibrées en ensembles d'entraînement et de test
    train_df, test_df = train_test_split(df_balanced, test_size=0.3,
                                         random_state=42)

    # divise le jeu d'entraînement en sous-ensemble pour la validation
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42
    )

    # dffiche la distribution des classes dans les ens. d'entraînement et test
    print("\nDistribution des classes dans l'ensemble d'entraînement :")
    print(train_df['label'].value_counts())
    print("\nDistribution des classes dans l'ensemble de test :")
    print(test_df['label'].value_counts())

    # extrait les textes et labels pour entraînement, validation et test
    train_texts = train_df['no_punct'].tolist()
    train_labels = train_df['label'].tolist()

    val_texts = val_df['no_punct'].tolist()
    val_labels = val_df['label'].tolist()

    test_texts = test_df['no_punct'].tolist()
    test_labels = test_df['label'].tolist()

    # charge le tokenizer DistilBERT pour convertir les textes en tokens
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # tokenise les textes pour l'entraînement, la validation et le test
    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=128
    )
    val_encodings = tokenizer(
        val_texts, truncation=True, padding=True, max_length=128
    )
    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, max_length=128
    )

    # crée datasets personnalisés pour entraînement, validation et test avec
    train_dataset = RedditDataset(train_encodings, train_labels) # encodage
    val_dataset = RedditDataset(val_encodings, val_labels)  # et labels
    test_dataset = RedditDataset(test_encodings, test_labels)

    # charge le modèle DistilBERT pour une classification binaire
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    )
    # déplace le modèle sur l'appareil disponible (CPU ou GPU)
    model.to(device)

    # configure les arguments pour l'entraînement du modèle
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, 'results'),  # chemin spécifié
        num_train_epochs=4,  # nombre d'époques d'entraînement
        per_device_train_batch_size=30,  # taille des lots pour l'entraînement
        per_device_eval_batch_size=30,  # taille des lots pour l'évaluation
        warmup_steps=300,  # nb d'étapes pour le warmup du taux d'apprentissage
        weight_decay=0.01,  # taux de pénalité de pondération
        logging_dir=os.path.join(args.output_dir, 'logs'),  # dossier des logs
        logging_steps=10,  # fréquence de journalisation des étapes
        eval_strategy='epoch',  # stratégie d'évaluation après chaque époque
        learning_rate=4e-5,  # taux d'apprentissage
        save_strategy="epoch",  # sauvegarde du modèle après chaque époque
        save_total_limit=2  # limite le nombre de sauvegardes
    )

    # initialise le Trainer avec le modèle, les arguments et les datasets
    trainer = Trainer(
        model=model,   # modèle DistilBERT
        args=training_args,  # arguments d'entraînement
        train_dataset=train_dataset,  # dataset pour l'entraînement
        eval_dataset=val_dataset,  # dataset pour la validation
    )

    # lance l'entraînement du modèle
    trainer.train()

    # évalue le modèle sur l'ensemble de validation
    val_results = trainer.evaluate(val_dataset)
    print("\n=== Résultats sur l'ensemble de validation ===")
    print(val_results)   # affiche les métriques d'évaluation

    # évalue le modèle sur l'ensemble de test
    test_results = trainer.evaluate(test_dataset)
    print("\n=== Résultats sur l'ensemble de test ===")
    print(test_results)   # affiche les métriques sur l'ensemble de test

    # sauvegarde le modèle et le tokenizer entraînés
    model.save_pretrained(
        os.path.join(args.output_dir, 'fine_tuned_distilbert'))
    tokenizer.save_pretrained(
        os.path.join(args.output_dir, 'fine_tuned_distilbert'))

    # affiche des exemples de prédictions sur les données de test
    print("\n=== Exemples de prédictions ===")
    with torch.no_grad():  # désact. calcul gradients pour accélérer inférence
        # tokenise les textes de test pour l'inférence
        test_texts_tokenized = tokenizer(
            test_texts, truncation=True, padding=True,
            max_length=128, return_tensors='pt'
        ).to(device)
        # génère les prédictions à partir du modèle
        test_outputs = model(**test_texts_tokenized)
        # détermine les labels prédits
        test_predictions = torch.argmax(
            test_outputs.logits, dim=-1
        ).cpu().numpy()

    # boucle sur quelques échantillons pour afficher les prédictions
    for i in range(5):
        print(f"Texte : {test_texts[i]}\nPrédiction : "
              f"{'POSITIVE' if test_predictions[i] == 1 else 'NEGATIVE'}\n")

    # affiche un rapport de classification complet sur l'ensemble de test
    print("\n=== Rapport de classification sur l'ensemble de test ===")
    print(
        classification_report(
            test_labels, test_predictions,  # labels réels et prédictions
            target_names=['NEGATIVE', 'POSITIVE']  # noms des classes
        ))
    print("\n\n")

# point d'entrée principal pour exécuter le script
if __name__ == "__main__":
    """
    Point d'entrée du script pour effectuer l'analyse de sentiment sur les
    tables de la base de données PostgreSQL.
    """
    # arguments de test par défaut pour exécuter le script pendant dévlp.
    #sys.argv = [
        #'6_my_fine_model.py',
        #'--tables',   # spécifie les tables à analyser
        #'posts_table_month_controversial_11_15',
        #'posts_table_month_top_11_15',
        #'posts_table_year_controversial_11_15',
        #'posts_table_year_top_11_15',
        #'comments_table_month_controversial_11_15',
        #'comments_table_month_top_11_15',
        #'comments_table_year_controversial_11_15',
        #'comments_table_year_top_11_15',
        #'--output_dir', '../test/fine_tuned/fine_tuned_comments'
    #]
    main()  # exécute la fonction principale
