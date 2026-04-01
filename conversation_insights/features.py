from __future__ import annotations

from conversation_insights.models import (
    ConversationFeatureRecord,
    ConversationMeta,
    DerivedMetrics,
    GroupedConversationRecord,
    LLMReview,
    StructureMetrics,
)
from conversation_insights.text_utils import (
    extract_links,
    extract_product_names,
    is_login_link,
    is_product_link,
)


def extract_conversation_features(grouped_records: list[GroupedConversationRecord]) -> list[ConversationFeatureRecord]:
    return [extract_feature_record(record) for record in grouped_records]


def extract_feature_record(record: GroupedConversationRecord) -> ConversationFeatureRecord:
    user_text_messages = [m.cleanText for m in record.messages if m.sender == "user" and m.messageType == "text" and m.cleanText]
    agent_text_messages = [m.cleanText for m in record.messages if m.sender == "agent" and m.messageType == "text" and m.cleanText]
    event_messages = [m for m in record.messages if m.messageType == "event"]

    event_types = sorted({message.eventType for message in event_messages if message.eventType})
    link_click_count = 0
    product_view_count = 0
    product_click_count = 0
    feedback_click_count = 0
    login_click_count = 0
    has_whatsapp_handoff = False

    recommended_product_names: list[str] = []
    for text in agent_text_messages:
        recommended_product_names.extend(extract_product_names(text))

    contains_product_links = False
    for text in agent_text_messages:
        links = extract_links(text)
        if any(is_product_link(link) for link in links):
            contains_product_links = True

    for message in event_messages:
        raw_text = message.rawText.strip()
        lowered = raw_text.lower()
        links = extract_links(raw_text)

        if message.eventType == "link_click" or raw_text.startswith("Clicked link:"):
            link_click_count += 1
        if message.eventType == "product_view" or raw_text.startswith("Viewed product:"):
            product_view_count += 1
        if message.eventType == "similar_product_click" or raw_text.startswith("Requested similar products to"):
            product_click_count += 1
        if message.eventType == "feedback_click" or "feedback" in lowered:
            feedback_click_count += 1
        if "whatsapp" in lowered:
            has_whatsapp_handoff = True

        for link in links:
            if is_login_link(link):
                login_click_count += 1

        recommended_product_names.extend(extract_product_names(raw_text))

    recommended_product_names = list(dict.fromkeys(name for name in recommended_product_names if name))
    product_click_count += product_view_count

    structure = StructureMetrics(
        num_messages=len(record.messages),
        num_event_messages=len(event_messages),
        num_user_turns=len(user_text_messages),
        num_agent_turns=len(agent_text_messages),
    )
    conversation_meta = ConversationMeta(
        eventTypes=event_types,
        linkClickCount=link_click_count,
        productClickCount=product_click_count,
        feedbackClickCount=feedback_click_count,
        productViewCount=product_view_count,
        loginClickCount=login_click_count,
        hasWhatsAppHandoff=has_whatsapp_handoff,
        containsProductLinks=contains_product_links,
        numRecommendationsObserved=len(recommended_product_names),
        recommendedProductNames=recommended_product_names,
    )

    return ConversationFeatureRecord(
        conversationId=record.conversationId,
        widgetId=record.widgetId,
        brandName=record.brandName,
        structure=structure,
        conversationMeta=conversation_meta,
        llmReview=LLMReview(),
        derivedMetrics=DerivedMetrics(),
    )
