#!/bin/bash
# =============================================================================
# Script d'installation et de configuration des dépendances Python et
# PostgreSQL
# Date: 2024-12-21
# Description: Ce script installe les dépendances Python, configure NLTK et
#              spaCy et installe PostgreSQL selon le système d'exploitation
#              détecté (mac et Linux compatibles). Nécessite Python3 & pip3!
# =============================================================================

set -e
# arrête le script en cas d'erreur

# affiche un message d'information
echo_info() {
    echo "[INFO] $1"
    # affiche un message d'information avec le préfixe [INFO]
}

# affiche un message de succès
echo_success() {
    echo "[SUCCESS] $1"
    # affiche un message de succès avec le préfixe [SUCCESS]
}

# affiche un message d'erreur et quitte le script
echo_error() {
    echo "[ERROR] $1"
    # affiche un message d'erreur avec le préfixe [ERROR]
    exit 1
    # quitte le script avec un statut d'erreur
}

# se déplace dans le répertoire où se trouve le script
cd "$(dirname "$0")"

# vérifie si python3 est installé
if ! command -v python3 &> /dev/null
then
    echo_error "Python3 n'est pas installé. Veuillez l'installer avant de continuer."
    # affiche une erreur si python3 n'est pas trouvé et quitte
fi

# vérifie si pip3 est installé
if ! command -v pip3 &> /dev/null
then
    echo_error "Pip3 n'est pas installé. Veuillez l'installer avant de continuer."
    # affiche une erreur si pip3 n'est pas trouvé et quitte
fi

# met à jour pip
echo_info "Mise à jour de pip..."
pip3 install --upgrade pip -qq
if [ $? -ne 0 ]; then
    echo_error "Échec de la mise à jour de pip."
    # affiche une erreur en cas d'échec de mise à jour
fi

# affiche un succès après la mise à jour
echo_success "Pip mis à jour."

# fonction d'installation de postgresql pour macos
install_postgres_mac() {
    if ! command -v brew &> /dev/null; then
        echo_error "Homebrew non installé. Installe-le depuis https://brew.sh/"
        # affiche une erreur si homebrew n'est pas installé
    fi

    echo_info "Mise à jour de Homebrew et installation de PostgreSQL 14..."
    brew update -qq
    brew uninstall --ignore-dependencies libpq -qq || true
    brew install libpq postgresql@14 -qq
    # met à jour homebrew et installe libpq et postgresql 14
    echo_info "Configuration et démarrage de PostgreSQL..."
    brew services restart postgresql@14
    # redémarre les services postgresql

    export PATH="/usr/local/opt/postgresql@14/bin:$PATH"
    # met à jour le path pour inclure postgresql
}

# fonction d'installation de postgresql pour ubuntu/debian
install_postgres_debian() {
    echo_info "Suppression des anciennes versions de PostgreSQL..."
    sudo apt-get -y remove --purge postgresql* || true
    sudo apt-get -y update -qq
    # supprime les anciennes versions et met à jour les paquets

    echo_info "Installation de PostgreSQL 14..."
    sudo apt-get -y install postgresql-14 libpq-dev -qq
    sudo systemctl enable postgresql
    sudo systemctl start postgresql
    # installe postgresql 14 et active son service
}

# détecte le système d'exploitation
OS=""
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
else
    echo_error "Système d'exploitation non supporté : $OSTYPE"
    # affiche une erreur si l'os n'est pas pris en charge
fi

# affiche l'os détecté
echo_info "Système d'exploitation détecté : $OS"

# installe postgresql selon l'os
echo_info "Installation de PostgreSQL..."
if [ "$OS" == "linux" ]; then
    install_postgres_debian
elif [ "$OS" == "mac" ]; then
    install_postgres_mac
else
    echo_error "Système non supporté pour PostgreSQL."
    # affiche une erreur si l'os n'est pas compatible
fi

# vérifie si le fichier requirements.txt existe, sinon le crée
if [ ! -f "requirements.txt" ]; then
    echo_info "Création du fichier requirements.txt..."
    cat > requirements.txt <<EOL
sqlalchemy>=2.0.36
psycopg2-binary>=2.9.6
sentencepiece>=0.1.97
pandas>=2.2.3
scikit-learn>=1.5.2
nltk>=3.9.1
praw>=7.8.1
transformers>=4.46.3
torch>=2.5.1
tqdm>=4.67.1
matplotlib>=3.9.3
wordcloud>=1.9.4
spacy>=3.8.2
accelerate>=0.26.0
urllib3<2.0
EOL
    # crée un fichier requirements.txt avec les dépendances nécessaires
    echo_success "Fichier requirements.txt créé."
else
    echo_info "Fichier requirements.txt déjà existant."
    # informe que le fichier requirements.txt existe déjà
fi

# installe les dépendances python
echo_info "Installation des dépendances Python..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo_error "Échec de l'installation des dépendances Python."
    # affiche une erreur si l'installation échoue
fi

# affiche un succès après l'installation des dépendances
echo_success "Dépendances Python installées."

# télécharge les ressources nltk
echo_info "Téléchargement des ressources nltk (stopwords, punkt)..."
# indique le début du téléchargement des ressources nltk
python3 -m nltk.downloader stopwords punkt
# télécharge les ressources stopwords et punkt pour nltk
if [ $? -ne 0 ]; then
    echo_error "Échec du téléchargement des ressources nltk."
    # affiche une erreur si le téléchargement échoue et quitte
fi
# fin du téléchargement des ressources nltk
echo_success "Ressources nltk téléchargées."
# confirme que les ressources nltk ont été téléchargées

# télécharge le modèle anglais de spacy
echo_info "Téléchargement du modèle anglais de spacy (en_core_web_sm)..."
# indique le début du téléchargement du modèle spacy
python3 -m spacy download en_core_web_sm
# télécharge le modèle linguistique en_core_web_sm pour spacy
if [ $? -ne 0 ]; then
    echo_error "Échec du téléchargement du modèle spacy."
    # affiche une erreur si le téléchargement échoue et quitte
fi
# fin du téléchargement du modèle spacy
echo_success "Modèle spacy téléchargé."
# confirme que le modèle spacy a été téléchargé

# affiche un succès après l'installation de postgresql
echo_success "PostgreSQL installé."
