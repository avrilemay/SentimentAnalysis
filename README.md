# SentimentAnalysis

# Analyse des Sentiments sur Reddit - Élections Présidentielles Américaines 2024

## Description du Projet

Ce projet a pour objectif d'explorer les sentiments exprimés sur Reddit en lien avec les élections présidentielles américaines de 2024. En utilisant des techniques de fouille de données et des outils de traitement du langage naturel (NLP), nous avons analysé les posts et les commentaires du subreddit [r/politics](https://www.reddit.com/r/politics/), axé sur la politique américaine. Ce projet permet de capter les émotions et opinions d'une partie de la population active sur Reddit concernant cet événement politique majeur.

### Motivations

L'année 2024 a été marquée par de nombreux événements politiques significatifs, dont les élections présidentielles américaines. Ces élections ont suscité des discussions intenses en ligne, notamment sur Reddit. L'objectif principal était d'analyser les sentiments (positifs ou négatifs) exprimés dans les posts et commentaires relatifs à cet événement afin de mieux comprendre les émotions d'une population ciblée. Ce projet n'a pas pour ambition d'être représentatif des opinions globales, mais d'offrir un aperçu unique des dynamiques émotionnelles sur cette plateforme.

---

## Structure du Projet
Le projet est organisé comme suit :

```plaintext
projet/
├── data/
├── sorties/
└── src/
```

- **data** : Contient les données brutes ou pré-traitées utilisées dans le projet.
- **sorties** : Contient les résultats et sorties générés par le projet.
- **src** : Contient le code source du projet.

---

## Guide d'Installation et d'Exécution

### Prérequis

1. **Système d'exploitation recommandé** :
   - macOS ou Debian/Ubuntu

2. **Python, Pip et Brew** :
   - Assurez-vous que **Python**, **pip3**, et **Brew** (pour macOS) sont installés.
   - Note : La version Python **3.14** n'est pas compatible avec certaines bibliothèques.

#### Vérification des versions
Exécutez les commandes suivantes pour confirmer les versions installées :

```bash
python3 --version
pip3 --version
brew --version
```

---

### Configuration de l'Environnement

1. Ajoutez le chemin d'accès correct à votre terminal (exemple pour macOS) :

```bash
export PATH="$PATH:/usr/local/bin"
```

2. Placez-vous dans le répertoire **src** :

```bash
cd src
```

3. Donnez les droits d'exécution aux scripts :

```bash
chmod +x install.sh tests.sh
```

---

### Installation

1. Placez-vous dans le répertoire **src** :

```bash
cd src
```

2. Lancez le script d'installation :

```bash
./install.sh
```

3. Vérifiez que l'installation s'est correctement terminée.

---

### Tests

1. **Avant d'exécuter les tests** : Assurez-vous que le répertoire `test` n'est pas encore présent.
   - Si ce répertoire n'existe pas, il sera créé pendant l'exécution des tests.

2. Placez-vous dans le répertoire **src** :

```bash
cd src
```

3. Exécutez le script de test :

```bash
./tests.sh
```

4. Après exécution, vérifiez que le répertoire `test` a été créé avec la structure attendue.

---

## Structure du Répertoire Après les Tests

Après exécution des tests, le répertoire `test` aura la structure suivante :

```plaintext
test/
├── fine_tuned/
│   ├── fine_tuned_comments/
│   │   ├── fine_tuned_distilbert/
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   ├── vocab.txt
│   │   ├── results/
│   │   │   ├── checkpoint-198/
│   │   │   │   ├── config.json
│   │   │   │   ├── model.safetensors
│   │   │   │   ├── optimizer.pt
│   │   │   │   ├── rng_state.pth
│   │   │   │   ├── scheduler.pt
│   │   │   │   ├── trainer_state.json
│   │   │   │   ├── training_args.bin
│   │   │   ├── checkpoint-264/
│   │   │   │   ├── config.json
│   │   │   │   ├── model.safetensors
│   │   │   │   ├── optimizer.pt
│   │   │   │   ├── rng_state.pth
│   │   │   │   ├── scheduler.pt
│   │   │   │   ├── trainer_state.json
│   │   │   │   ├── training_args.bin
│   ├── fine_tuned_posts/
│   │   ├── fine_tuned_distilbert/
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   ├── vocab.txt
│   │   ├── results/
│   │   │   ├── checkpoint-112/
│   │   │   │   ├── config.json
│   │   │   │   ├── model.safetensors
│   │   │   │   ├── optimizer.pt
│   │   │   │   ├── rng_state.pth
│   │   │   │   ├── scheduler.pt
│   │   │   │   ├── trainer_state.json
│   │   │   │   ├── training_args.bin
│   │   │   ├── checkpoint-84/
│   │   │   │   ├── config.json
│   │   │   │   ├── model.safetensors
│   │   │   │   ├── optimizer.pt
│   │   │   │   ├── rng_state.pth
│   │   │   │   ├── scheduler.pt
│   │   │   │   ├── trainer_state.json
│   │   │   │   ├── training_args.bin
├── manual_eval/
│   ├── positive_entries_posts_et_comments.csv
├── mock_data_today/
│   ├── manual_eval/
│   │   ├── positive_entries_posts_et_comments.csv
│   │   ├── shuffled_comments_year_top_today.csv
│   │   ├── shuffled_posts_year_top_today.csv
│   ├── raw/
│   │   ├── year_top_today/
│   │   │   ├── comments_year_top_today.csv
│   │   │   ├── posts_year_top_today.csv
│   ├── sorties/
│   │   ├── models_perfs_top_today/
│   │   │   ├── perf_models_global.csv
│   │   │   ├── perf_models_posts.csv
├── sorties/
│   ├── emotion_repartition.csv
│   ├── graph_time/
│   │   ├── année_controversée_2023-11-15_2024-11-15.png
│   │   ├── mois_controversé_2024-10-15_2024-11-15.png
│   │   ├── année_top_2023-11-15_2024-11-15.png
│   │   ├── mois_top_2024-10-15_2024-11-15.png
│   ├── models_perfs_project_data/
│   │   ├── perf_models_comments.csv
│   │   ├── perf_models_global.csv
│   │   ├── perf_models_posts.csv
│   ├── tdidf_plots/
│   │   ├── negative_all.png
│   │   ├── positive_all.png
```
