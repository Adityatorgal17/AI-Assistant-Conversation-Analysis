from __future__ import annotations

import argparse
import json
from pathlib import Path

from conversation_insights.config import (
    DEFAULT_CONVERSATIONS_FILE,
    DEFAULT_MESSAGES_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MONGO_DB,
    MongoSettings,
)
from conversation_insights.etl import build_grouped_conversations
from conversation_insights.features import extract_conversation_features
from conversation_insights.insights import build_dashboard_rows, build_global_summary, build_widget_insights
from conversation_insights.llm_review import maybe_run_llm_reviews
from conversation_insights.mongo_store import write_processed_collections

GENERATED_OUTPUT_FILENAMES = (
    "grouped_conversations.json",
    "conversation_features.json",
    "dashboard_rows.json",
    "widget_insights.json",
    "global_summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conversation insights and dashboard-ready outputs.")
    parser.add_argument("--conversations-file", type=Path, default=DEFAULT_CONVERSATIONS_FILE)
    parser.add_argument("--messages-file", type=Path, default=DEFAULT_MESSAGES_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mongo-uri", type=str, default=None)
    parser.add_argument("--mongo-db", type=str, default=DEFAULT_MONGO_DB)
    parser.add_argument("--write-mongo", action="store_true")
    parser.add_argument("--clear-llm-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clear_generated_outputs(args.output_dir)
    if args.clear_llm_cache:
        clear_llm_cache(args.output_dir)

    grouped_records = build_grouped_conversations(args.conversations_file, args.messages_file)
    feature_records = extract_conversation_features(grouped_records)
    feature_records = maybe_run_llm_reviews(
        grouped_records=grouped_records,
        feature_records=feature_records,
        output_dir=args.output_dir,
    )
    dashboard_rows = build_dashboard_rows(feature_records)
    widget_insights = build_widget_insights(feature_records)
    global_summary = build_global_summary(feature_records)

    grouped_dicts = [record.to_dict() for record in grouped_records]
    feature_dicts = [record.to_dict() for record in feature_records]
    dashboard_dicts = [row.to_dict() for row in dashboard_rows]
    widget_dicts = [summary.to_dict() for summary in widget_insights]
    global_dict = global_summary.to_dict()

    write_json(args.output_dir / "grouped_conversations.json", grouped_dicts)
    write_json(args.output_dir / "conversation_features.json", feature_dicts)
    write_json(args.output_dir / "dashboard_rows.json", dashboard_dicts)
    write_json(args.output_dir / "widget_insights.json", widget_dicts)
    write_json(args.output_dir / "global_summary.json", global_dict)

    if args.write_mongo:
        if not args.mongo_uri:
            raise RuntimeError("--write-mongo was set, but no --mongo-uri was provided.")
        settings = MongoSettings(uri=args.mongo_uri, database=args.mongo_db)
        write_processed_collections(
            settings=settings,
            grouped_records=grouped_dicts,
            feature_records=feature_dicts,
            dashboard_rows=dashboard_dicts,
            widget_insights=widget_dicts,
            global_summary=global_dict,
        )

    print(f"Processed {len(grouped_records)} conversations into {args.output_dir}")


def write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=True)


def clear_generated_outputs(output_dir: Path) -> None:
    for filename in GENERATED_OUTPUT_FILENAMES:
        path = output_dir / filename
        if path.exists() and path.is_file():
            path.unlink()


def clear_llm_cache(output_dir: Path) -> None:
    path = output_dir / "llm_review_cache.json"
    if path.exists() and path.is_file():
        path.unlink()


if __name__ == "__main__":
    main()
