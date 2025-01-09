"""
0_set_up_db.py

Script de configuration d'une base de données PostgreSQL

Fonctionnalités :
- Supprime une base existante si nécessaire
- Vérifie ou crée un rôle PostgreSQL avec mot de passe
- Crée une base de données associée au rôle spécifié
- Restaure les données depuis un fichier dump binaire

Exemple :
    python3 0_set_up_db.py [fichier_dump] -- ou "reddit_db.dump" par défaut
"""

import os  # gestion des fichiers et chemins
import platform  # détection du système d'exploitation
import subprocess  # exécution des commandes système
import sys  # pour gérer les arguments de script
import shutil  # pour copier les fichiers



def drop_database_if_exists(database_name, sudo_prefix, sudo_cwd):
    """
    Supprime une base de données PostgreSQL si elle existe.

    Arguments :
        database_name (str) : Nom de la base à supprimer
        sudo_prefix (str) : Préfixe sudo selon le système (vide ou
                            'sudo -u postgres' pour Linux)
        cwd (str) : Répertoire de travail pour la commande
    """
    try:
        # vérifie si la base existe
        print(
            f"Vérification si la base '{database_name}' existe pour "
            f"suppression..."
        )
        check_db_cmd = ( # définit la commande pour vérifier si la bdd existe
            f"{sudo_prefix}psql -d postgres -tAc "  # -tAc pour obtenir "1"
            f"\"SELECT 1 FROM pg_database WHERE datname='{database_name}'\""
        )   # retourne 1 si la bdd existe

        result = subprocess.run(   # lance la commande pour vérif existence db
            # utilisation du shell, récupère la sortie au format str
            check_db_cmd, shell=True, stdout=subprocess.PIPE,
            # répertoire de travail actuel, erreur sur /dev/null
            text=True, cwd=sudo_cwd, stderr=subprocess.DEVNULL
        )
        if result.stdout.strip() == '1':  # base existe (sortie = "1")
            # supprime la base
            print(f"Suppression de la base '{database_name}'...")
            # création de la commande (avec/sans préfixe)
            drop_db_cmd = f"{sudo_prefix}dropdb {database_name}"
            subprocess.run(drop_db_cmd, shell=True, check=True, cwd=sudo_cwd)
            # utilisation du shell dans rép travail + vérif succès check
            print(f"Base '{database_name}' supprimée avec succès.")

        else:   # si la base n'existe pas (la sortie =/= "1")
            print(  # pas besoin de supprimer
                f"La base '{database_name}' n'existe pas. "
                "Aucune suppression nécessaire."
            )

    except subprocess.CalledProcessError as e:
        # erreur lors de la suppression de la bdd (issue du "check=True")
        print(
            f"Erreur lors de la suppression de la base '{database_name}' : {e}"
        )
        raise


def ensure_role_exists(role_name, sudo_prefix, role_password, sudo_cwd):
    """
    Vérifie et crée ou met à jour un rôle PostgreSQL.

    Arguments :
        role_name (str) : Nom du rôle à vérifier/créer
        sudo_prefix (str) : Préfixe sudo selon le système (vide ou
                            'sudo -u postgres' pour Linux)
        role_password (str) : Mot de passe à attribuer au rôle
        cwd (str) : Répertoire de travail pour la commande
    """
    try:
        # vérifie si le rôle existe
        print(f"Vérification si le rôle '{role_name}' existe...")
        check_role_cmd = ( # création de la commande pour vérification
            f"{sudo_prefix}psql -d postgres -tAc " # -tAc pour "1" en sortie
            f"\"SELECT 1 FROM pg_roles WHERE rolname='{role_name}'\""
        )    # retourne "1" si le rôle existe

        result = subprocess.run(  # exécution de la commande
            check_role_cmd, shell=True, stdout=subprocess.PIPE,
            # utilisation du shell, sortie sur stdout au format str
            text=True, cwd=sudo_cwd, stderr=subprocess.DEVNULL
            # répertoire accès défini + erreur sur /dev/null
        )

        if result.stdout.strip() != '1':   # si le rôle n'existe pas (=/= "1")
            # crée le rôle avec mot de passe
            print(f"Création du rôle '{role_name}' avec mot de passe...")
            create_role_cmd = (   # création de la commande avec nom et pw
                f"{sudo_prefix}psql -d postgres -c "  # "-c" cmd sql depuis CLI
                # création du rôle avec le mdp 
                f"\"CREATE ROLE {role_name} WITH LOGIN PASSWORD "
                f"'{role_password}'\""
            )
            # lancement de la commande pour création du rôle avec mdp
            subprocess.run(  # shell, vérif de la réussite, rép. accès défini
                create_role_cmd, shell=True, check=True, cwd=sudo_cwd
            )
            print(f"Rôle '{role_name}' créé avec succès.")   # création réussie

        else:   # si le rôle existe déjà
            print(  # on va mettre à jour le mdp
                f"Le rôle '{role_name}' existe déjà. "
                "Mise à jour du mot de passe..."
            )
            alter_role_cmd = (   # création commande pour màj du mdp
                f"{sudo_prefix}psql -d postgres -c "
                f"\"ALTER ROLE {role_name} WITH PASSWORD "
                f"'{role_password}'\""
            )
            subprocess.run(  # exécution de la cmd pour modifier le mdf
                alter_role_cmd, shell=True, check=True, cwd=sudo_cwd
                # usage terminal dans rep. travail et vérif. succès
            )
            print(  # message de confirmation de màj du mdp
                f"Mot de passe du rôle '{role_name}' mis à jour avec "
                "succès."
            )
    except subprocess.CalledProcessError as e:  # erreur (check=True)
        print(   # erreur gestion du rôle (création ou modif du mdp)
            f"Erreur lors de la création ou mise à jour du rôle "
            f"'{role_name}' : {e}"
        )
        raise


def setup_database(dump_file="dump_id_clean.dump"):
    """
    Configure une base de données PostgreSQL : suppression existante,
    création, restauration

    Arguments :
        dump_file (str) : Nom du fichier de sauvegarde pour la
                          restauration (situé dans le même répertoire
                          que le script)
    """
    try:   # recherche l'OS utilisé
        system = platform.system()
        if system == "Darwin":  # macOS
            print("\n\nSystème détecté : macOS")
            sudo_prefix = ""  # pas de sudo pour PostgreSQL
            sudo_cwd = os.path.expanduser("~")  # répertoire personnel
        elif system == "Linux":  # Linux
            print("\n\nSystème détecté : Linux")
            sudo_prefix = "sudo -u postgres "   # besoin du sudo
            sudo_cwd = "/tmp"  # répertoire accessible par postgres
        else:
            raise EnvironmentError(  # autre système non supporté
                "Système non pris en charge. Fonctionne sur macOS et Linux."
            )

        # paramètres de la base et rôle
        database_name = "projet_reddit"   # nom db
        role_name = "avrile"  # utilisateur
        role_password = "projet"  # mdp pour l'utilisateur

        # répertoire de travail = répertoire du script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Répertoire de travail défini sur : {script_dir}")

        # chemin absolu vers le dump (qui se trouve dans le répertoire)
        dump_file_path = os.path.join(script_dir, dump_file)

        # vérifie si le fichier dump existe
        if not os.path.exists(dump_file_path):
            print(   # si le fichier dump n'existe pas
                f"Fichier dump '{dump_file_path}' introuvable. "
                "Vérifiez le chemin."
            )
            sys.exit(1)   # erreur

        # copie le dump dans /tmp pour que postgres puisse y accéder
        accessible_dump = "/tmp/dump_id_clean.dump"
        try:
            shutil.copy(dump_file_path, accessible_dump)  # copie le dump
            print(
                f"Fichier dump copié vers '{accessible_dump}' pour "
                f"restauration.")
        except PermissionError as e:
            print(  # si la copie n'a pas marché : erreur
                f"Erreur : Impossible de copier le fichier dump '"
                f"{dump_file_path}' vers '{accessible_dump}'. Vérifiez les "
                f"permissions. {e}")
            sys.exit(1)   # fin du script

        # vérifie les permissions du fichier copié
        if not os.access(accessible_dump, os.R_OK):
            print(  # si pas de permission en lecture, donne erreur et solution
                f"Le fichier copié '{accessible_dump}' n'est pas lisible par "
                "l'utilisateur 'postgres'. Assurez-vous que les permissions "
                "sont correctes."
            )
            sys.exit(1)   # fin du script

        # supprime la base si elle existe
        drop_database_if_exists(database_name, sudo_prefix, sudo_cwd)

        # vérifie ou crée le rôle
        ensure_role_exists(role_name, sudo_prefix, role_password, sudo_cwd)

        # crée la base de données
        print(f"Création de la base '{database_name}'...")
        create_db_cmd = (  # commande pour créer la bdd
            f"{sudo_prefix}createdb -O {role_name} {database_name}"
        )
        subprocess.run(   # exéc cmd (shell + vérif. succès + rép. accès défini)
            create_db_cmd, shell=True, check=True, cwd=sudo_cwd
        )
        print(f"Base '{database_name}' créée avec succès.")  # succès

        # restaure la base depuis le dump accessible
        print(
            f"Restauration de la base '{database_name}' depuis '"
            f"{accessible_dump}'..."
        )

        # cmd pour restaurer depuis un fichier .dump binaire
        restore_cmd = (
            f"{sudo_prefix}pg_restore --dbname={database_name} "
            f"--clean --if-exists \"{accessible_dump}\" 2> /dev/null"
            # nettoie avant de restaurer (supprime les objets existants)
        )

        # exécute la commande de restauration
        subprocess.run( # usage du shell, vérif. succès, rép. accès défini
            restore_cmd, shell=True, check=True, cwd=sudo_cwd
        )
        print("Base restaurée avec succès.\n\n")


    except subprocess.CalledProcessError as e:   # erreur (check=True)
        print(f"Erreur lors de l'exécution : {e}\n\n")
        sys.exit(1)
    except Exception as e:     # erreur inattendue
        print(f"Erreur inattendue : {e}\n\n")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:    # possibilité de préciser le dump à utiliser
        dump_file_arg = sys.argv[1]
    else:   # sinon dump utilisé par défaut
        dump_file_arg = "dump_db.dump"

    # lancement de la config de la bdd
    setup_database(dump_file=dump_file_arg)
