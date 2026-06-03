# GPT Response Similarity Study

This repository contains a cleaned, reproducible version of the GPT response similarity analysis.

## Study Workflow

1. **Collect GPT responses manually**
   - GPT responses were collected manually outside this codebase.
   - The retained raw analysis inputs are the already-exploded sentence-level files:
     - `exploded_data_filtered2.csv`
     - `exploded_data_stages1.csv`
     - `exploded_data_stages2.csv`
     - `exploded_data_stages3.csv`

2. **Clean data**
   - Removes empty sentence rows.
   - Normalizes non-ASCII characters and repeated whitespace.
   - Keeps `ChatGpt 4o` rows for the across-user analysis.

3. **Run TF-IDF word similarity**
   - Combines sentences within each user/prompt or stage/category.
   - Computes cosine similarity on TF-IDF vectors.

4. **Run DeBERTa sentence similarity**
   - Uses `potsawee/deberta-v3-large-mnli`.
   - Computes directional entailment-probability matrices for each sentence-pair set.
   - Stores both directions so a symmetric sentence similarity matrix can be built.

5. **Aggregate sentence similarity with Hungarian alignment**
   - Symmetrizes DeBERTa matrices as `0.5 * (forward + backward.T)`.
   - Uses Hungarian alignment to match sentences and retain matched-pair similarity scores.

6. **Create visualizations and bootstrap mean differences**
   - Creates strip plots.
   - Runs nested mean-difference bootstraps:
     - Stage analysis: average sentence similarities within `User x StagePair`, compare paired user-level means, resample users.
     - User analysis: average sentence similarities within `user_pair x new_index`, compare paired user-pair-level means, resample user pairs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU/HPC runs, install the `torch` build that matches your CUDA environment.

## Run the Pipeline

From this folder:

```bash
export PYTHONPATH="$PWD/src"

# 1. Clean data
python -m gpt_similarity.clean_data

# 2. TF-IDF word similarity
python -m gpt_similarity.tfidf_similarity

# 3. DeBERTa sentence similarity
python -m gpt_similarity.deberta_similarity users --max-index 12 --output data/interim/deberta_user_matrices.pkl
python -m gpt_similarity.deberta_similarity stages --user-id 1
python -m gpt_similarity.deberta_similarity stages --user-id 2
python -m gpt_similarity.deberta_similarity stages --user-id 3

# 4. Hungarian aggregation
python -m gpt_similarity.aggregate_similarity users
python -m gpt_similarity.aggregate_similarity stages

# 5. Figures
python -m gpt_similarity.visualize users --input results/tables/sentence_similarity_users.csv
python -m gpt_similarity.visualize stages --input results/tables/sentence_similarity_stages.csv

# 6. Nested bootstrap mean differences
python -m gpt_similarity.bootstrap_stats users
python -m gpt_similarity.bootstrap_stats stages
```

The DeBERTa step is the slow step and may require GPU/HPC resources.
