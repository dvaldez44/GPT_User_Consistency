#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python -m gpt_similarity.clean_data
python -m gpt_similarity.tfidf_similarity

python -m gpt_similarity.deberta_similarity users --max-index 12 --output data/interim/deberta_user_matrices.pkl
python -m gpt_similarity.deberta_similarity stages --user-id 1
python -m gpt_similarity.deberta_similarity stages --user-id 2
python -m gpt_similarity.deberta_similarity stages --user-id 3

python -m gpt_similarity.aggregate_similarity users
python -m gpt_similarity.aggregate_similarity stages

python -m gpt_similarity.visualize users --input results/tables/sentence_similarity_users.csv
python -m gpt_similarity.visualize stages --input results/tables/sentence_similarity_stages.csv

python -m gpt_similarity.bootstrap_stats users
python -m gpt_similarity.bootstrap_stats stages
