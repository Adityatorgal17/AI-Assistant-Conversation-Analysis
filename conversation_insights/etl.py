from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from conversation_insights.models import GroupedConversationRecord, MessageRecord
from conversation_insights.text_utils import clean_agent_text, extract_links, infer_brand_names_by_widget


def load_json_array(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_grouped_conversations(
    conversations_path: Path,
    messages_path: Path,
) -> list[GroupedConversationRecord]:
    conversations = load_json_array(conversations_path)
    messages = load_json_array(messages_path)

    messages_by_conversation: dict[str, list[MessageRecord]] = defaultdict(list)
    widget_links: dict[str, list[str]] = defaultdict(list)
    conversation_to_widget = {conversation["_id"]: conversation["widgetId"] for conversation in conversations}

    for raw_message in messages:
        conversation_id = raw_message["conversationId"]
        raw_text = raw_message.get("text", "")
        clean_text = clean_agent_text(raw_text) if raw_message.get("sender") == "agent" else raw_text.strip()
        message = MessageRecord(
            messageId=raw_message["_id"],
            sender=raw_message["sender"],
            messageType=raw_message["messageType"],
            rawText=raw_text,
            cleanText=clean_text,
            eventType=(raw_message.get("metadata") or {}).get("eventType"),
            timestamp=raw_message["timestamp"],
        )
        messages_by_conversation[conversation_id].append(message)

        widget_id = conversation_to_widget.get(conversation_id)
        if widget_id:
            widget_links[widget_id].extend(extract_links(raw_text))

    inferred_brand_names = infer_brand_names_by_widget(widget_links)
    grouped_records: list[GroupedConversationRecord] = []

    for raw_conversation in conversations:
        conversation_id = raw_conversation["_id"]
        sorted_messages = sorted(
            messages_by_conversation[conversation_id],
            key=lambda item: item.timestamp,
        )
        grouped_records.append(
            GroupedConversationRecord(
                conversationId=conversation_id,
                widgetId=raw_conversation["widgetId"],
                brandName=inferred_brand_names.get(raw_conversation["widgetId"]),
                createdAt=raw_conversation["createdAt"],
                updatedAt=raw_conversation["updatedAt"],
                messages=sorted_messages,
            )
        )

    return grouped_records

