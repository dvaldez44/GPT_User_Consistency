from __future__ import annotations

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="GPT response similarity study pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Create exploded sentence files from manually collected answer spreadsheets.")
    sub.add_parser("clean", help="Clean manually collected GPT response files.")
    sub.add_parser("tfidf", help="Run TF-IDF cosine similarity analyses.")
    args = parser.parse_args()

    if args.command == "prepare":
        from .prepare_data import create_exploded_files

        create_exploded_files()
    elif args.command == "clean":
        from .clean_data import write_clean_outputs

        write_clean_outputs()
    elif args.command == "tfidf":
        from .tfidf_similarity import write_tfidf_outputs

        write_tfidf_outputs()


if __name__ == "__main__":
    main()
