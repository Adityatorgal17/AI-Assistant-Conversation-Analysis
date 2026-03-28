from __future__ import annotations

from typing import Any

from conversation_insights.config import MongoSettings

try:
    from pymongo import MongoClient
except ImportError:  # pragma: no cover - optional dependency
    MongoClient = None


def write_processed_collections(
    settings: MongoSettings,
    grouped_records: list[dict[str, Any]],
    feature_records: list[dict[str, Any]],
    dashboard_rows: list[dict[str, Any]],
    widget_insights: list[dict[str, Any]],
    global_summary: dict[str, Any],
) -> None:
    if MongoClient is None:
        raise RuntimeError(
            "pymongo is not installed. Install dependencies from "
            "`conversation_insights/requirements.txt` before writing to MongoDB."
        )

    client = MongoClient(settings.uri)
    database = client[settings.database]

    _replace_collection(database[settings.grouped_collection], grouped_records, "conversationId")
    _replace_collection(database[settings.features_collection], feature_records, "conversationId")
    _replace_collection(database[settings.dashboard_collection], dashboard_rows, "conversationId")
    _replace_collection(database[settings.widget_insights_collection], widget_insights, "widgetId")
    global_item = dict(global_summary)
    global_item["_id"] = "global_summary"
    database[settings.global_summary_collection].delete_many({})
    database[settings.global_summary_collection].insert_one(global_item)


def _replace_collection(collection: Any, documents: list[dict[str, Any]], id_field: str) -> None:
    collection.delete_many({})
    if not documents:
        return

    normalized = []
    for document in documents:
        item = dict(document)
        item["_id"] = item[id_field]
        normalized.append(item)
    collection.insert_many(normalized)
