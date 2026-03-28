from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from conversation_insights.config import DEFAULT_OUTPUT_DIR, MongoSettings

try:
    from pymongo import MongoClient
except ImportError:  # pragma: no cover - optional in UI mode
    MongoClient = None


st.set_page_config(
    page_title="Conversation Insights Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


def main() -> None:
    st.title("Conversation Insights Dashboard")
    st.caption("Per-widget and per-conversation analysis built from processed conversation records.")

    source = st.sidebar.radio("Data source", ["JSON outputs", "MongoDB"], index=0)
    if source == "JSON outputs":
        data = load_from_json(DEFAULT_OUTPUT_DIR)
    else:
        mongo_uri = st.sidebar.text_input("Mongo URI", value="mongodb://localhost:27017/")
        mongo_db = st.sidebar.text_input("Mongo DB", value="helio_intern")
        data = load_from_mongo(MongoSettings(uri=mongo_uri, database=mongo_db))

    if not data["dashboard_rows"]:
        st.error("No dashboard rows were found. Run the processing pipeline first.")
        return

    widget_filter = build_filters(data["dashboard_rows"])
    render_global_overview(data["global_summary"], widget_filter["filtered_rows"])
    st.divider()
    render_widget_insights(data["widget_insights"], widget_filter["selected_widget_id"])
    st.divider()
    render_conversation_table(widget_filter["filtered_rows"], widget_filter["widget_lookup"])
    st.divider()
    render_conversation_detail(
        dashboard_rows=widget_filter["filtered_rows"],
        features=data["conversation_features"],
        grouped=data["grouped_conversations"],
        widget_lookup=widget_filter["widget_lookup"],
    )


def build_filters(dashboard_rows: list[dict[str, Any]]) -> dict[str, Any]:
    widget_options = build_widget_options(dashboard_rows)
    widget_labels = ["All"] + [item["label"] for item in widget_options]
    widget_lookup = {item["widgetId"]: item["label"] for item in widget_options}
    label_to_id = {item["label"]: item["widgetId"] for item in widget_options}

    qualities = sorted({row["quality"] for row in dashboard_rows})
    outcomes = sorted({row["conversationOutcome"] for row in dashboard_rows})
    intents = sorted({row["initialIntent"] for row in dashboard_rows})

    selected_widget_label = st.sidebar.selectbox("Widget", options=widget_labels, index=0)
    selected_widget_id = None if selected_widget_label == "All" else label_to_id[selected_widget_label]
    selected_qualities = st.sidebar.multiselect("Quality", options=qualities, default=qualities)
    selected_outcomes = st.sidebar.multiselect("Outcome", options=outcomes, default=outcomes)
    selected_intents = st.sidebar.multiselect("Intent", options=intents, default=intents)

    filtered_rows = []
    for row in dashboard_rows:
        if selected_widget_id and row["widgetId"] != selected_widget_id:
            continue
        if row["quality"] not in selected_qualities:
            continue
        if row["conversationOutcome"] not in selected_outcomes:
            continue
        if row["initialIntent"] not in selected_intents:
            continue
        filtered_rows.append(row)

    return {
        "selected_widget_id": selected_widget_id,
        "filtered_rows": filtered_rows,
        "widget_lookup": widget_lookup,
    }


def build_widget_options(dashboard_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen = set()
    options = []
    for row in dashboard_rows:
        widget_id = row["widgetId"]
        if widget_id in seen:
            continue
        seen.add(widget_id)
        brand_name = humanize_brand(row.get("brandName")) or widget_id
        options.append({"widgetId": widget_id, "brandName": brand_name})

    options = sorted(options, key=lambda item: item["widgetId"])
    for index, item in enumerate(options, start=1):
        item["label"] = f"Widget {index} - {item['brandName']}"
    return options


def render_global_overview(global_summary: dict[str, Any], filtered_rows: list[dict[str, Any]]) -> None:
    st.subheader("Overview")

    filtered_count = len(filtered_rows)
    bad_count = sum(row["quality"] == "bad" for row in filtered_rows)
    unresolved_count = sum(row["unresolved"] for row in filtered_rows)
    recommendation_count = sum(row["recommendationGiven"] for row in filtered_rows)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filtered Conversations", filtered_count)
    col2.metric("Bad Quality", bad_count)
    col3.metric("Unresolved", unresolved_count)
    col4.metric("Recommendation Flows", recommendation_count)

    with st.expander("Global Summary", expanded=True):
        if global_summary:
            st.json(global_summary)
        else:
            st.info("Global summary is not available.")


def render_widget_insights(widget_insights: list[dict[str, Any]], selected_widget_id: str | None) -> None:
    st.subheader("Widget Insights")

    shown = widget_insights
    if selected_widget_id:
        shown = [item for item in widget_insights if item["widgetId"] == selected_widget_id]

    if not shown:
        st.info("No widget insight records matched the current filters.")
        return

    for insight in shown:
        label = humanize_brand(insight.get("brandName")) or insight["widgetId"]
        with st.container(border=True):
            st.markdown(f"### {label}")
            st.caption(f"Widget ID: `{insight['widgetId']}`")

            col1, col2, col3 = st.columns(3)
            col1.metric("Conversations", insight["totalConversations"])
            col2.metric("Bad Quality", insight["qualityBreakdown"].get("bad", 0))
            col3.metric("Unresolved", insight["outcomeBreakdown"].get("unresolved", 0))

            if insight["summaryPoints"]:
                for point in insight["summaryPoints"]:
                    st.write(f"- {point}")

            left, right = st.columns(2)
            with left:
                st.markdown("**Top Intents**")
                st.table(insight["topIntents"])
            with right:
                st.markdown("**Top Problems**")
                st.table(insight["topProblems"] or [{"problem": "none", "count": 0}])


def render_conversation_table(filtered_rows: list[dict[str, Any]], widget_lookup: dict[str, str]) -> None:
    st.subheader("Conversation Table")
    if not filtered_rows:
        st.info("No conversations matched the current filters.")
        return

    table_rows = []
    for row in filtered_rows:
        table_rows.append(
            {
                "widget": widget_lookup.get(row["widgetId"], row["widgetId"]),
                "conversationId": row["conversationId"],
                "intent": row["initialIntent"],
                "quality": row["quality"],
                "score": row["score"],
                "outcome": row["conversationOutcome"],
                "success": row["success"],
                "unresolved": row["unresolved"],
                "badRecommendation": row["badRecommendation"],
                "possibleClaimRisk": row["possibleClaimRisk"],
                "messages": row["numMessages"],
            }
        )

    st.dataframe(table_rows, use_container_width=True, hide_index=True)


def render_conversation_detail(
    dashboard_rows: list[dict[str, Any]],
    features: list[dict[str, Any]],
    grouped: list[dict[str, Any]],
    widget_lookup: dict[str, str],
) -> None:
    st.subheader("Conversation Detail")

    if not dashboard_rows:
        st.info("No filtered conversations available.")
        return

    conversation_labels = [
        f"{row['conversationId']} | {widget_lookup.get(row['widgetId'], row['widgetId'])} | {row['conversationOutcome']}"
        for row in dashboard_rows
    ]
    label_to_id = {label: row["conversationId"] for label, row in zip(conversation_labels, dashboard_rows)}
    selected_label = st.selectbox("Conversation", options=conversation_labels)
    selected_conversation_id = label_to_id[selected_label]

    dashboard_row = next(row for row in dashboard_rows if row["conversationId"] == selected_conversation_id)
    feature_row = next((row for row in features if row["conversationId"] == selected_conversation_id), None)
    grouped_row = next((row for row in grouped if row["conversationId"] == selected_conversation_id), None)

    meta_left, meta_right = st.columns(2)
    with meta_left:
        st.markdown("**Conversation Meta**")
        st.json(dashboard_row)
    with meta_right:
        st.markdown("**Feature Record**")
        st.json(feature_row or {})

    st.markdown("**Transcript**")
    if not grouped_row:
        st.info("Grouped conversation record not available.")
        return

    for message in grouped_row["messages"]:
        speaker = message["sender"].upper()
        badge = f"`{message['messageType']}`"
        st.markdown(f"**{speaker}** {badge}  \n{message['cleanText'] or message['rawText']}")
        st.caption(message["timestamp"])


@st.cache_data(show_spinner=False)
def load_from_json(output_dir: Path) -> dict[str, Any]:
    return {
        "grouped_conversations": read_json(output_dir / "grouped_conversations.json", []),
        "conversation_features": read_json(output_dir / "conversation_features.json", []),
        "dashboard_rows": read_json(output_dir / "dashboard_rows.json", []),
        "widget_insights": read_json(output_dir / "widget_insights.json", []),
        "global_summary": read_json(output_dir / "global_summary.json", {}),
    }


@st.cache_data(show_spinner=False)
def load_from_mongo(settings: MongoSettings) -> dict[str, Any]:
    if MongoClient is None:
        st.error("pymongo is not installed. Install dependencies first.")
        return empty_data()

    try:
        client = MongoClient(settings.uri, serverSelectionTimeoutMS=3000)
        database = client[settings.database]
        client.admin.command("ping")
    except Exception as exc:  # pragma: no cover - runtime connectivity path
        st.error(f"MongoDB connection failed: {exc}")
        return empty_data()

    global_summary = database[settings.global_summary_collection].find_one({"_id": "global_summary"}) or {}
    global_summary.pop("_id", None)

    return {
        "grouped_conversations": list(database[settings.grouped_collection].find({}, {"_id": 0})),
        "conversation_features": list(database[settings.features_collection].find({}, {"_id": 0})),
        "dashboard_rows": list(database[settings.dashboard_collection].find({}, {"_id": 0})),
        "widget_insights": list(database[settings.widget_insights_collection].find({}, {"_id": 0})),
        "global_summary": global_summary,
    }


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def empty_data() -> dict[str, Any]:
    return {
        "grouped_conversations": [],
        "conversation_features": [],
        "dashboard_rows": [],
        "widget_insights": [],
        "global_summary": {},
    }


def humanize_brand(value: str | None) -> str | None:
    if not value:
        return value
    value = value.replace("_", " ").replace("-", " ").strip()
    return value.title()


if __name__ == "__main__":
    main()
