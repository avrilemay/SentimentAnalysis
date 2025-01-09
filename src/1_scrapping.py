"""
1_scrapping.py

Ce script collecte des données Reddit (posts et commentaires) via l'API PRAW,
les filtre selon des critères, les stocke dans une base PostgreSQL, puis les
exporte en CSV.

Fonctionnalités :
- Extraction de posts/commentaires depuis Reddit
- Filtrage par mots-clés, période et tri
- Stockage dans une base PostgreSQL avec création des tables
- Export des données au format CSV (pour référence)

Exemple :
    python script.py --folders ../data  [folder pour raw data]
                     --post_filename posts.csv [nom du csv pour les posts]
                     --comment_filename comments.csv [nom du csv pour les coms]
                     --post_table posts_table [nom table pour les posts]
                     --comment_table comments_table [nom table pour les coms]
                     --subreddits politics [subredit]
                     --keywords trump,biden [mots-clés à rechercher]
                     --time_filter year [year / month / week / all]
                     --sort_by top [top / controversial] -- tri des posts
                     --total_posts 100 [combien de posts à récupérer, max=1000]
                     --comments_per_post 10 [combioen de coms par posts]
"""

import argparse  # options en CLI
import sys  # arguments en CLI
import praw  # API Reddit pour extraire des données
import os  # opérations sur les fichiers et répertoires
import pandas as pd  # manipulation de données
from datetime import datetime  # gestion des dates et heures

from sqlalchemy import (  # import pour la bdd
    create_engine, MetaData, Table, Column, Integer, String,
    Text, ForeignKey, DateTime, text, inspect
)
from sqlalchemy.exc import OperationalError   # pb de connexion
from sqlalchemy.dialects.postgresql import insert as pg_insert  # on conflict
import configparser  # pour lire le fichier de configuration


# configuration des arguments pour le script
parser = argparse.ArgumentParser(description="Extraction de données Reddit")
# chemin du dossier où sauvegarder les fichiers:
parser.add_argument('--folders', type=str, help='Dossier de sortie')
# nom du fichier CSV pour sauvegarder les posts extraits:
parser.add_argument('--post_filename', type=str, help='Csv des posts')
# nom du fichier CSV pour sauvegarder les commentaires extraits:
parser.add_argument('--comment_filename', type=str,
                    help='Csv des commentaires')
# nom de la table PostgreSQL pour stocker les posts:
parser.add_argument('--post_table', type=str, help='Table des posts')
# nom de la table PostgreSQL pour stocker les commentaires:
parser.add_argument('--comment_table', type=str,
                    help='Table des commentaires')
# période des posts à récupérer : heure, jour, semaine, mois, année, ou tout:
parser.add_argument('--time_filter', type=str,
                    choices=['hour', 'day', 'week', 'month', 'year', 'all'],
                    help='Période de collecte')
# critère de tri des posts : top, nouveautés, controverses, etc.:
parser.add_argument('--sort_by', type=str,
                    choices=['hot', 'new', 'top', 'controversial', 'rising'],
                    help='Critère de tri des posts')
# liste des subreddits ciblés, séparés par des virgules:
parser.add_argument('--subreddits', type=str,
                    help='Subreddits ciblés, séparés par des virgules')
# liste des mots-clés à rechercher dans les titres ou les commentaires:
parser.add_argument('--keywords', type=str,
                    help='Mots-clés, séparés par des virgules')
# nombre total de posts à récupérer par subreddit:
parser.add_argument('--total_posts', type=int,
                    help='Nombre de posts à extraire par subreddit')
# nombre de commentaires à récupérer par post:
parser.add_argument('--comments_per_post', type=int,
                    help='Nombre de commentaires à récupérer par post')

# définition de l'URL de connexion
DATABASE_URL = "postgresql://avrile:projet@localhost:5432/projet_reddit"

def connect_db():
    """
    Connexion à la base de données PostgreSQL.

    Returns:
        sqlalchemy.engine.Engine: Objet de connexion à la base de données
    """
    try:
        engine = create_engine(DATABASE_URL, isolation_level="AUTOCOMMIT")
        # vérifie la connexion
        with engine.connect() as conn:
            conn.execute(text("SELECT 1")) # exéc d'une commande simple
        return engine
    except OperationalError as e:  # si la connexion ne marche pas
        print(f"Impossible de se connecter à la base de données : {e}")
        raise  # erreur


def create_tables():
    """
    Crée les tables dans la base de données si elles n'existent pas
    Arrête le script si les tables existent déjà
    """
    engine = connect_db()  # connexion à la db
    inspector = inspect(engine)  # pour véfifier la structure de la bdd

    # vérifie si la table des posts existe
    if inspector.has_table(args.post_table):
        print(f"\nLa table '{args.post_table}' existe déjà. Arrêt du script.\n\n")
        exit()  # arrêt si la table des posts existe

    # vérifie si la table des commentaires existe
    if inspector.has_table(args.comment_table):
        print(f"\nLa table '{args.comment_table}' existe déjà. Arrêt du script.\n\n")
        exit()  # arrêt si la table des commentaires existe déjà

    # définition des tables (qui n'existent pas)
    metadata = MetaData()

    # définition de la table des posts
    posts_table = Table(   # nom d'après l'arg, associe la table à métadata
        args.post_table, metadata,  # ID unique:
        Column('post_id', String, primary_key=True, nullable=False),
        Column('type', String),  # type -post
        Column('title', Text),  # titre du post
        Column('score', Integer),  # score du post
        Column('num_comments', Integer),  # nombre de commentaires
        Column('created', DateTime),  # date de création
        Column('subreddit', String),  # nom du subreddit
        Column('url', String),  # URL du post
        Column('content', Text)  # contenu du post
    )

    # définition de la table des commentaires
    comments_table = Table(   # nom d'après l'arg, associe la table à métadata
        args.comment_table, metadata,
        Column('comment_id', String, primary_key=True,
               nullable=False),  # ID du comm unique, ne peut pas être nul
        Column('post_id', String, ForeignKey(f'{args.post_table}.post_id'),
               nullable=False),  # ID unique du post associé  (clé étrangère)
        Column('type', String),  # type - commentaire
        Column('content', Text),  # contenu du commentaire
        Column('score', Integer),   # score du commentaire
        Column('created', DateTime),   # date de création
    )

    # création des tables dans la bdd via metadata
    metadata.create_all(engine)
    print(f"\n\nTable '{args.post_table}' créée.")
    print(f"Table '{args.comment_table}' créée avec une clé étrangère.")


def store_data():
    """
    Récupère les posts et commentaires depuis Reddit et les insère dans la
    base de données
    """
    engine = connect_db()      # récupère connexion à postgreSQL
    metadata = MetaData()   # création objet MetaData
    metadata.reflect(bind=engine)  # récupère les tables existantes de la bdd
    posts_table = metadata.tables[args.post_table]  # get réf à table des posts
    comments_table = metadata.tables[args.comment_table]  # réf à table comms
    connection = engine.connect()        # connexion à la bdd

    # sépare la liste des subreddits fournie par des virgules et la nettoie
    subreddits = [sub.strip() for sub in args.subreddits.split(',')]

    for subreddit_name in subreddits:   # boucle sur chaque subreddit
        print(f"Traitement du subreddit : {subreddit_name}")

        # récupère le subreddit via l'API Reddit
        subreddit = reddit.subreddit(subreddit_name)
        posts_collected = 0  # initialisation compteur de posts collectés

        # sélection des posts selon le critère de tri choisi par l'utilisateur
        if args.sort_by == 'top':
            # posts les plus populaires sur une période donnée
            posts = subreddit.top(time_filter=args.time_filter,
                                  limit=total_posts_per_subreddit)
        elif args.sort_by == 'controversial':  # posts les plus controversés
            posts = subreddit.controversial(time_filter=args.time_filter,
                                            limit=total_posts_per_subreddit)
        elif args.sort_by == 'hot':  # posts populaires actuellement
            posts = subreddit.hot(limit=total_posts_per_subreddit)
        elif args.sort_by == 'new':   # posts récents
            posts = subreddit.new(limit=total_posts_per_subreddit)
        elif args.sort_by == 'rising':  # posts qui gagnent en popularité
            posts = subreddit.rising(limit=total_posts_per_subreddit)
        else:
            # par défaut, top
            posts = subreddit.top(time_filter=args.time_filter,
                                  limit=total_posts_per_subreddit)

        # parcourt les posts récupérés
        for post in posts:
            # arrête la colecte si la limite par subreddit est atteinte
            if posts_collected >= total_posts_per_subreddit:
                break  # limite atteinte

            # filtre les posts par mots-clés et exclusion des "megathreads"
            if (any(keyword in post.title.lower() for keyword in keywords) and
                    "megathread" not in post.title.lower()):
                # prépare une requête d'insertion pour la table des posts
                stmt = pg_insert(posts_table).values(
                    post_id=post.id,
                    type="post",    # type = post
                    title=post.title,   # titre du post
                    score=post.score,      # score du post
                    num_comments=post.num_comments,     # nb de commentaires
                    created=datetime.fromtimestamp(post.created_utc),   # créa.
                    subreddit=subreddit_name,   # nom du subreddit
                    url=post.url,       # URL du post
                    content=post.selftext       # contenu du post
                ).on_conflict_do_nothing()  # évite dupli. si post existe déjà
                connection.execute(stmt)        # exécute l'insertion

                posts_collected += 1  # incrémente le compteur

                # récupère les commentaires du post dans la limite définie
                post.comments.replace_more(limit=replace_more_limit)
                comments = post.comments.list()   # liste complète des comms
                print(  # statut de la collecte
                    f"Post ID : {post.id}, Titre : {post.title[:50]}..., "
                    f"Commentaires récupérés : {min(len(comments), comments_per_post)}"
                )

                # trie les commentaires par score (en partant du + élevé)
                sorted_comments = sorted(
                    comments, key=lambda comment: comment.score, reverse=True
                )
                # récupère seulement le nombre de commentaire souhaité
                relevant_comments = sorted_comments[:comments_per_post]

                # insère chaque commentaire dans la base de données
                for comment in relevant_comments:
                    # vérifie qu'il s'agit d'un commentaire
                    if (isinstance(comment, praw.models.Comment) and
                            # avec un ID et un contenu non supprimé
                            comment.id is not None and
                            comment.body not in ["[deleted]", "[removed]"] and
                            # et qui contient un des mots-clés recherchés
                            any(keyword in comment.body.lower()
                                for keyword in keywords)):

                        # préparation de la requête d'insertion dans la bdd
                        stmt = pg_insert(comments_table).values(
                            comment_id=comment.id,      # id du commentaire
                            post_id=post.id,        # id du post
                            type="comment",         # type - commentaire
                            content=comment.body,       # contenu du com
                            score=comment.score,        # score et date
                            created=datetime.fromtimestamp(comment.created_utc)
                        ).on_conflict_do_nothing()  # évite la duplication

                        # exécute la requête d'insertion dans la base
                        connection.execute(stmt)

    connection.close()  # ferme la connexion


def get_posts_and_comments():
    """
    Récupère les posts et commentaires depuis la base et les retourne sous
    forme de DataFrames

    Returns:
        tuple: Deux pandas.DataFrame, l'un avec les posts et l'autre les coms
    """
    engine = connect_db()   # connexion PostgreSQL
    # extrait tous les posts depuis la table des posts sous forme de DF
    posts_df = pd.read_sql_table(args.post_table, engine)
    # extrait tous les commentaires depuis la table des comm sous forme de DF
    comments_df = pd.read_sql_table(args.comment_table, engine)
    # supprime les doublons dans les données extraites
    posts_df = posts_df.drop_duplicates(subset='post_id').reset_index(drop=True)
    comments_df = comments_df.drop_duplicates(
        subset='comment_id').reset_index(drop=True)
    # retourne les deux DataFrames
    return posts_df, comments_df



if __name__ == "__main__":
    """
    Point d'entrée du script pour collecter et stocker des données Reddit
    """
    # liste des arguments pour la phase de développement
    #sys.argv = [
    #    '1_scrapping.py',
    #    '--folders', '../test/data/raw/year_top_today',  # folder sauvegarde csv
    #    '--post_filename', 'posts_year_top_today.csv',  # csv post
    #    '--comment_filename', 'comments_year_top_today.csv',  # csv comment
    #    '--post_table', 'posts_table_year_top_today',   # table post
    #    '--comment_table', 'comments_table_year_top_today',   # table comment
    #    '--time_filter', 'year',    # filtre temporel
    #    '--sort_by', 'top',    # classé par
    #    '--subreddits', 'politics',    # subreddit étudié
    #    '--keywords', (   # mots clés recherchés
    #        'trump,harris,donald,president,campaign,democrats,'
    #        'republicans,ballot,swing state,polls,kamala,primary,'
    #        'vance,walz,election,elections,maga,jd'
    #    ),
    #    '--total_posts', '10',  # limite pour les posts
    #    '--comments_per_post', '5'  # limite pour les commentaires
    #]

    # analyse des arguments via argparse
    args = parser.parse_args()   # rend les arg accessible via args

    # liste des mots-clés en minuscules, séparés et nettoyés
    keywords = [kw.strip().lower() for kw in args.keywords.split(',')]
    # nombre total de posts à récupérer par subreddit
    total_posts_per_subreddit = args.total_posts
    # nombre de commentaires par post
    comments_per_post = args.comments_per_post
    # limite pour charger les sous-commentaires
    replace_more_limit = 4

    # configuration des chemins de sortie d'après les args
    output_folder = args.folders  # dossier de sauvegarde des fichiers CSV
    os.makedirs(output_folder, exist_ok=True)  # crée le dossier si pas existant
    # chemin complet pour le fichier des posts
    post_file_path = os.path.join(output_folder, args.post_filename)
    # chemin complet pour le fichier des commentaires
    comment_file_path = os.path.join(output_folder, args.comment_filename)

    # lecture du fichier de configuration
    config = configparser.ConfigParser()
    config.read(  # fichier dans le même répertoire
        'config.ini')
    # récupération des informations Reddit depuis le fichier de configuration
    reddit_config = config['REDDIT']

    # initialisation de l'API Reddit avec PRAW
    reddit = praw.Reddit(
        client_id=reddit_config['client_id'],  # ID client
        client_secret=reddit_config['client_secret'],  # secret client
        user_agent=reddit_config['user_agent'],  # ID app
        username=reddit_config['username'],  # nom d'utilisateur Reddit
        password=reddit_config['password']  # mot de passe Reddit
    )

    # création des tables dans la base de données
    create_tables()
    # extraction des données depuis Reddit et stockage dans la base
    store_data()
    # chargement des données dans des DataFrames
    posts_df, comments_df = get_posts_and_comments()
    # export des données dans des fichiers CSV
    posts_df.to_csv(post_file_path, index=False)  # sauvegarde des posts
    comments_df.to_csv(comment_file_path, index=False)  # sauvegarde des coms
    # message de confirmation
    print(f"Les fichiers ont été sauvegardés dans le dossier {output_folder}\n\n")

