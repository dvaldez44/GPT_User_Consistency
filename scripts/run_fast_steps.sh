#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python -m gpt_similarity.clean_data
python -m gpt_similarity.tfidf_similarity

# These require existing DeBERTa matrix pickles in data/interim/.
python -m gpt_similarity.aggregate_similarity users
python -m gpt_similarity.aggregate_similarity stages
python -m gpt_similarity.visualize users --input results/tables/sentence_similarity_users.csv
python -m gpt_similarity.visualize stages --input results/tables/sentence_similarity_stages.csv
python -m gpt_similarity.bootstrap_stats users
python -m gpt_similarity.bootstrap_stats stages
