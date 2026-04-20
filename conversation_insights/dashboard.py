from __future__ import annotations

import html
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


GLOSSARY_TERMS: dict[str, str] = {
    "Escalation": "Hand-off or need for human/support intervention when the bot cannot resolve effectively.",
    "Unresolved": "Conversation ended without solving the user's core need.",
    "Resolved": "Conversation ended with the user's need solved.",
    "Drop Off": "User stopped engaging before the issue was resolved.",
    "Frustrated": "Conversation showed frustration or dissatisfaction signals.",
    "Recommendation Flows": "Conversations where product recommendations were given.",
    "Recommendation Given": "The assistant offered a product or next-step recommendation.",
    "Recommendation Relevant": "The recommendation matched the user’s need.",
    "Recommendation Converted": "The user acted on the recommendation.",
    "Bad Recommendation": "The assistant suggested something mismatched, misleading, or unhelpful.",
    "Escalation Needed": "Cases where human intervention should have happened.",
    "Escalation Triggered": "Cases where escalation was actually triggered.",
    "Success": "The conversation achieved the intended outcome.",
    "product_page_question": "User asks about product details, ingredients, usage, benefits, or suitability.",
    "order_support": "User needs help with order status, edits, cancellation, refund, delivery, or tracking.",
    "product_discovery": "User asks for suggestions or best-product guidance for a need.",
    "other": "Intent did not confidently match known categories.",
    "greeting": "Greeting or small-talk opener with no specific task yet.",
    "risk_flagged": "Potential compliance/safety risk detected in the conversation.",
    "containsPossibleClaimRisk": "The conversation included wording that may be interpreted as a claim or compliance risk.",
    "containsSafetyDisclaimer": "A safety or compliance disclaimer was present.",
    "order_friction": "Order-related journey had blockers, repeated steps, or missing guidance.",
    "login_loop": "Conversation got stuck in repeated login/sign-in prompts without progress.",
    "productClick": "A product click was observed in the conversation metadata.",
    "productView": "A product view was observed in the conversation metadata.",
    "linkClick": "A link click was observed in the conversation metadata.",
    "feedbackClick": "A feedback/action button was clicked in the conversation metadata.",
    "assistantRepetition": "The assistant repeated itself or got stuck in a loop.",
    "qualityDimensions": "Per-conversation quality scores across accuracy, relevance, clarity, helpfulness, tone, efficiency, and escalation handling.",
}

FLAG_STYLE_MAP = {
    "danger": {"bg": "#3b1717", "border": "#d26a6a", "text": "#ffb0b0"},
    "warning": {"bg": "#3b2e12", "border": "#d1a34d", "text": "#f4d58d"},
    "info": {"bg": "#132638", "border": "#4f86c6", "text": "#b4d4ff"},
}


def main() -> None:
    st.title("Conversation Insights Dashboard")
    st.caption("Global summary and widget-level deep dive from processed conversation records.")

    render_glossary_sidebar()

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

    widget_options = build_widget_options(data["dashboard_rows"])
    widget_lookup = {item["widgetId"]: item["label"] for item in widget_options}
    label_to_id = {item["label"]: item["widgetId"] for item in widget_options}

    view_mode = st.sidebar.radio("View", ["Global Summary", "Widget Insights"], index=0)

    if view_mode == "Global Summary":
        render_global_overview(data["global_summary"], data["dashboard_rows"])
        return

    if not widget_options:
        st.info("No widget data is available.")
        return

    selected_widget_label = st.sidebar.selectbox(
        "Select Widget",
        options=[item["label"] for item in widget_options],
        index=0,
    )
    selected_widget_id = label_to_id[selected_widget_label]
    widget_rows = [row for row in data["dashboard_rows"] if row["widgetId"] == selected_widget_id]

    render_widget_insights(data["widget_insights"], selected_widget_id)
    st.divider()
    render_conversation_table(widget_rows, widget_lookup)
    st.divider()
    render_conversation_detail(
        dashboard_rows=widget_rows,
        features=data["conversation_features"],
        grouped=data["grouped_conversations"],
        widget_lookup=widget_lookup,
    )


def render_glossary_sidebar() -> None:
    with st.sidebar.expander("Dashboard Glossary", expanded=False):
        for term, definition in GLOSSARY_TERMS.items():
            st.markdown(f"**{term}**: {definition}")


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


def render_global_overview(global_summary: dict[str, Any], all_rows: list[dict[str, Any]]) -> None:
    st.subheader("Global Summary")

    total_count = len(all_rows)
    bad_count = sum(row["quality"] == "bad" for row in all_rows)
    unresolved_count = sum(row["unresolved"] for row in all_rows)
    recommendation_count = sum(row["recommendationGiven"] for row in all_rows)
    escalation_needed = sum(row.get("escalationNeeded", False) for row in all_rows)
    escalation_triggered = sum(row.get("escalationTriggered", False) for row in all_rows)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Conversations", total_count)
    col2.metric("Bad Quality", bad_count)
    col3.metric("Unresolved", unresolved_count)
    col4.metric("Recommendation Flows", recommendation_count)
    col5.metric("Escalation Needed", escalation_needed)
    col6.metric("Escalation Triggered", escalation_triggered)

    if global_summary:
        recommendations = global_summary.get("recommendations", [])
        if recommendations:
            st.markdown("**Global Recommendations**")
            render_recommendations(recommendations, key_prefix="global")

        with st.expander("Global Summary JSON", expanded=False):
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

            if insight.get("qualityDimensionsAggregate"):
                st.markdown("**Quality Dimension Averages (-2 to +2)**")
                st.json(insight["qualityDimensionsAggregate"])

            recommendations = insight.get("recommendations", [])
            st.markdown("**Recommendations**")
            if not recommendations:
                st.info("No recommendations available for this widget.")
            else:
                render_recommendations(recommendations, key_prefix=insight["widgetId"])

            with st.expander("Mistake Counters", expanded=False):
                left, right = st.columns(2)
                with left:
                    st.markdown("**Assistant Mistakes**")
                    st.json(insight.get("assistantMistakes", {}))
                with right:
                    st.markdown("**User Mistakes**")
                    st.json(insight.get("userMistakes", {}))


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
                "accuracy": row.get("qualityAccuracy", 0),
                "relevance": row.get("qualityRelevance", 0),
                "clarity": row.get("qualityClarity", 0),
                "helpfulness": row.get("qualityHelpfulness", 0),
                "tone": row.get("qualityTone", 0),
                "efficiency": row.get("qualityEfficiency", 0),
                "escalationNeeded": row.get("escalationNeeded", False),
                "escalationTriggered": row.get("escalationTriggered", False),
                "escalationResolved": row.get("escalationResolved", False),
                "messages": row["numMessages"],
            }
        )

    st.dataframe(table_rows, use_container_width=True, hide_index=True)


def render_recommendations(recommendations: list[dict[str, Any]], key_prefix: str) -> None:
    for idx, rec in enumerate(recommendations, start=1):
        title = rec.get("title", "Untitled Recommendation")
        with st.expander(f"{idx}. {title}", expanded=(idx == 1)):
            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
            meta_col1.metric("Severity", str(rec.get("severity", "unknown")).upper())
            meta_col2.metric("Related Count", rec.get("relatedCount", 0))
            percentage_value = rec.get("percentage")
            if isinstance(percentage_value, (int, float)):
                meta_col3.metric("Affected", f"{percentage_value:.1f}%")
            else:
                meta_col3.metric("Affected", "N/A")

            source = str(rec.get("source", "deterministic"))
            meta_col4.metric("Source", source)

            confidence = rec.get("confidence")
            if isinstance(confidence, (int, float)):
                st.caption(f"Confidence: {confidence:.2f}")

            why_new = rec.get("whyNew")
            if why_new:
                st.markdown(f"**Why New**: {why_new}")

            description = rec.get("description")
            if description:
                st.write(description)

            causes_col, actions_col = st.columns(2)
            with causes_col:
                st.markdown("**Root Causes**")
                causes = rec.get("rootCauses", [])
                if causes:
                    for cause in causes:
                        st.write(f"- {cause}")
                else:
                    st.write("- Not available")

            with actions_col:
                st.markdown("**Suggested Actions**")
                actions = rec.get("suggestedActions", [])
                if actions:
                    for action in actions:
                        st.write(f"- {action}")
                else:
                    st.write("- Not available")

            evidence = rec.get("evidenceMetrics", {})
            if isinstance(evidence, dict) and evidence:
                with st.expander("Evidence Metrics", expanded=False):
                    st.json(evidence)


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

    message_flags = build_message_flag_lookup(feature_row, grouped_row)
    flagged_messages = [(index, flags) for index, flags in message_flags.items() if flags]

    if flagged_messages:
        st.markdown("**Flagged Message Signals**")
        for index, flags in flagged_messages:
            message = grouped_row["messages"][index]
            sender = str(message.get("sender", "")).upper()
            labels = ", ".join(flag["label"] for flag in flags)
            st.caption(f"{sender} | {message.get('timestamp', '')} | {labels}")
    else:
        st.caption("No LLM message flags were returned for this conversation.")

    # Chat-like layout: user left, assistant right
    for index, message in enumerate(grouped_row["messages"]):
        sender = message["sender"].lower()
        text = (message["cleanText"] or message["rawText"]).strip()
        timestamp = message["timestamp"]
        message_type = message["messageType"]
        flags = message_flags.get(index, [])

        if sender == "user":
            left_col, right_col = st.columns([1, 3])
            with left_col:
                st.markdown(f"**USER** `{message_type}`")
                st.caption(timestamp)
                render_message_flag_badges(flags)
            with right_col:
                render_message_bubble(text, bubble_color="#1976d2", align="left")
                render_message_flag_notes(flags, align="left")

        elif sender == "agent":
            left_col, right_col = st.columns([3, 1])
            with left_col:
                render_message_bubble(text, bubble_color="#7b1fa2", align="right")
                render_message_flag_notes(flags, align="right")
            with right_col:
                st.markdown(f"**AGENT** `{message_type}`")
                st.caption(timestamp)
                render_message_flag_badges(flags)

        st.divider()


def build_message_flag_lookup(
    feature_row: dict[str, Any] | None,
    grouped_row: dict[str, Any],
) -> dict[int, list[dict[str, str]]]:
    messages = grouped_row.get("messages", [])
    message_flags: dict[int, list[dict[str, str]]] = {index: [] for index in range(len(messages))}

    llm_review = ((feature_row or {}).get("llmReview") or {}) if isinstance(feature_row, dict) else {}
    raw_flags = llm_review.get("messageFlags", [])
    if not isinstance(raw_flags, list):
        return message_flags

    message_index_by_id = {
        str(message.get("messageId")): index
        for index, message in enumerate(messages)
        if message.get("messageId") is not None
    }

    for raw_flag in raw_flags:
        if not isinstance(raw_flag, dict):
            continue
        message_id = str(raw_flag.get("messageId") or "").strip()
        message_index = message_index_by_id.get(message_id)
        if message_index is None:
            continue

        label = str(raw_flag.get("label") or humanize_flag(str(raw_flag.get("flag") or ""))).strip()
        reason = str(raw_flag.get("reason") or "").strip()
        severity = str(raw_flag.get("severity") or "medium").strip().lower()
        message_flags[message_index].append(
            {
                "label": label or "Flagged Message",
                "note": reason or label or "LLM flagged this message for reviewer attention.",
                "tone": tone_for_flag(severity, str(raw_flag.get("flag") or "")),
            }
        )

    return message_flags


def render_message_bubble(text: str, bubble_color: str, align: str) -> None:
    text_html = html.escape(text).replace("\n", "<br>")
    justify = "flex-start" if align == "left" else "flex-end"
    st.markdown(
        (
            f"<div style='display:flex; justify-content:{justify};'>"
            f"<div style='max-width: 92%; background-color: {bubble_color}; color: white; "
            "padding: 12px; border-radius: 8px; font-size: 14px; line-height: 1.5;'>"
            f"{text_html}"
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def render_message_flag_badges(flags: list[dict[str, str]]) -> None:
    if not flags:
        return

    badge_html = []
    for flag in flags:
        style = FLAG_STYLE_MAP.get(flag["tone"], FLAG_STYLE_MAP["info"])
        badge_html.append(
            f"<span style='display:inline-block; margin: 4px 6px 0 0; padding: 3px 8px; "
            f"border-radius: 999px; border: 1px solid {style['border']}; background: {style['bg']}; "
            f"color: {style['text']}; font-size: 11px; font-weight: 700; letter-spacing: 0.04em; "
            f"text-transform: uppercase;'>{html.escape(flag['label'])}</span>"
        )
    st.markdown("".join(badge_html), unsafe_allow_html=True)


def render_message_flag_notes(flags: list[dict[str, str]], align: str) -> None:
    if not flags:
        return

    note_html = []
    for flag in flags:
        style = FLAG_STYLE_MAP.get(flag["tone"], FLAG_STYLE_MAP["info"])
        note_html.append(
            f"<div style='margin-top: 6px; max-width: 92%; border-left: 3px solid {style['border']}; "
            f"padding: 8px 10px; color: {style['text']}; background: rgba(255,255,255,0.03); "
            f"font-size: 12px; border-radius: 6px;'>{html.escape(flag['note'])}</div>"
        )

    st.markdown(
        f"<div style='display:flex; flex-direction:column; align-items:{'flex-start' if align == 'left' else 'flex-end'};'>"
        + "".join(note_html)
        + "</div>",
        unsafe_allow_html=True,
    )


def tone_for_flag(severity: str, flag_name: str) -> str:
    normalized_flag = flag_name.strip().lower()
    if severity == "high":
        return "danger"
    if normalized_flag in {"assistant_claim_risk", "user_frustrated", "assistant_login_loop"}:
        return "danger"
    if severity == "medium":
        return "warning"
    return "info"


def humanize_flag(flag_name: str) -> str:
    return flag_name.replace("_", " ").strip().title() or "Flagged Message"


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
