from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_CONVERSATIONS_FILE = PROJECT_ROOT / "conversations.json"
DEFAULT_MESSAGES_FILE = PROJECT_ROOT / "messages.json"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"
DEFAULT_MONGO_DB = "helio_intern"


@dataclass(slots=True)
class MongoSettings:
    uri: str
    database: str = DEFAULT_MONGO_DB
    grouped_collection: str = "processed_grouped_conversations"
    features_collection: str = "processed_conversation_features"
    dashboard_collection: str = "processed_dashboard_rows"
    widget_insights_collection: str = "processed_widget_insights"
    global_summary_collection: str = "processed_global_summary"
