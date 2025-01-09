"""
0_test_conn.py

Script pour tester la connexion à une base PostgreSQL.

Fonctionnalités :
- Vérifie la connexion à la base via SQLAlchemy
- Exécute une requête simple pour valider la connexion
- Affiche un message en cas de succès ou d'erreur

Exemple :
    python script.py
"""

from sqlalchemy import create_engine, text
# gestion des connexions et exécution des requêtes SQL


def test_connection():
    """
    Teste la connexion à une base de données PostgreSQL.

    Retourne :
        sqlalchemy.engine.Engine or None : Moteur SQLAlchemy si la
            connexion réussit, sinon None

    Exceptions gérées :
        Exception : Affiche un message d'erreur en cas de problème
            de connexion
    """
    try:
        # création du moteur SQLAlchemy avec les paramètres de connexion
        engine = create_engine(
            "postgresql://avrile:projet@localhost:5432/projet_reddit"
        )

        # tentative de connexion à la base de données
        with engine.connect() as connection:
            # requête simple pour vérifier la connexion
            result = connection.execute(
                text("SELECT 1")
            )  # renvoie 1 si la connexion fonctionne
            print("\n\nConnexion réussie :", result.scalar())  # affiche le
            # résultat

        # retourne le moteur en cas de succès
        return engine

    except Exception as e:
        # affiche un message d'erreur si la connexion échoue
        print("\n\nErreur de connexion :", e)
        return None


if __name__ == "__main__":
    """
    Point d'entrée du script pour tester la connexion à la base de données
    """
    # test de connexion à la base de données
    engine = test_connection()

    # vérifie si la connexion a réussi
    if engine:
        print("Connexion établie avec succès.\n\n")
    else:
        print("Échec de la connexion.\n\n")
