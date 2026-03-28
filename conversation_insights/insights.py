from __future__ import annotations

from collections import Counter, defaultdict

from conversation_insights.models import (
    ConversationFeatureRecord,
    DashboardConversationRow,
    GlobalInsightSummary,
    WidgetInsightSummary,
)


def build_dashboard_rows(records: list[ConversationFeatureRecord]) -> list[DashboardConversationRow]:
    rows: list[DashboardConversationRow] = []
    for record in records:
        llm = record.llmReview
        meta = record.conversationMeta
        rows.append(
            DashboardConversationRow(
                widgetId=record.widgetId,
                brandName=record.brandName,
                conversationId=record.conversationId,
                initialIntent=llm.initialIntent or "other",
                numMessages=record.structure.num_messages,
                numUserTurns=record.structure.num_user_turns,
                numAgentTurns=record.structure.num_agent_turns,
                whoEndedConversation=record.structure.who_ended_conversation,
                quality=record.derivedMetrics.quality,
                score=record.derivedMetrics.score,
                success=bool(llm.success),
                dropOff=bool(llm.dropOff),
                frustrated=bool(llm.frustrated),
                unresolved=bool(llm.unresolved),
                recommendationGiven=bool(llm.recommendationGiven),
                recommendationClicked=meta.productViewCount > 0,
                productClick=meta.productClickCount > 0,
                linkClick=meta.linkClickCount > 0,
                feedbackClick=meta.feedbackClickCount > 0,
                assistantRepetition=bool(llm.assistantRepetition),
                badRecommendation=bool(llm.badRecommendation),
                possibleClaimRisk=bool(llm.containsPossibleClaimRisk),
                conversationOutcome=llm.conversationOutcome or "neutral",
                primaryProblem=llm.primaryProblem,
                feedbackText=llm.feedbackText,
            )
        )
    return rows


def build_widget_insights(records: list[ConversationFeatureRecord]) -> list[WidgetInsightSummary]:
    widget_groups: dict[str, list[ConversationFeatureRecord]] = defaultdict(list)
    for record in records:
        widget_groups[record.widgetId].append(record)

    summaries: list[WidgetInsightSummary] = []
    for widget_id, widget_records in sorted(widget_groups.items()):
        summaries.append(_build_widget_summary(widget_id, widget_records))
    return summaries


def build_global_summary(records: list[ConversationFeatureRecord]) -> GlobalInsightSummary:
    quality_breakdown = Counter(record.derivedMetrics.quality for record in records)
    outcome_breakdown = Counter((record.llmReview.conversationOutcome or "neutral") for record in records)
    top_intents = Counter((record.llmReview.initialIntent or "other") for record in records).most_common(5)
    top_problems = Counter(record.llmReview.primaryProblem for record in records if record.llmReview.primaryProblem).most_common(5)

    summary_points = [
        f"{quality_breakdown.get('bad', 0)} conversations are currently scored as bad quality.",
        f"The most common intent is {top_intents[0][0]}." if top_intents else "No intent summary available.",
        f"{outcome_breakdown.get('unresolved', 0)} conversations are flagged as unresolved.",
    ]
    return GlobalInsightSummary(
        totalConversations=len(records),
        qualityBreakdown=dict(quality_breakdown),
        outcomeBreakdown=dict(outcome_breakdown),
        topIntents=[{"intent": intent, "count": count} for intent, count in top_intents],
        topProblems=[{"problem": problem, "count": count} for problem, count in top_problems],
        summaryPoints=summary_points,
    )


def _build_widget_summary(widget_id: str, records: list[ConversationFeatureRecord]) -> WidgetInsightSummary:
    brand_name = records[0].brandName if records else None
    quality_breakdown = Counter(record.derivedMetrics.quality for record in records)
    outcome_breakdown = Counter((record.llmReview.conversationOutcome or "neutral") for record in records)
    intent_breakdown = Counter((record.llmReview.initialIntent or "other") for record in records)
    problem_breakdown = Counter(record.llmReview.primaryProblem for record in records if record.llmReview.primaryProblem)

    recommendation_conversations = sum(bool(record.llmReview.recommendationGiven) for record in records)
    recommendation_clicks = sum(record.conversationMeta.productViewCount > 0 for record in records)
    unresolved_count = outcome_breakdown.get("unresolved", 0)
    login_loops = problem_breakdown.get("login_loop", 0)
    order_friction = problem_breakdown.get("order_friction", 0)

    summary_points: list[str] = []
    if records:
        summary_points.append(
            f"{quality_breakdown.get('bad', 0)} of {len(records)} conversations are scored as bad quality."
        )
    if intent_breakdown:
        top_intent, top_intent_count = intent_breakdown.most_common(1)[0]
        summary_points.append(f"Top intent is {top_intent} with {top_intent_count} conversations.")
    if unresolved_count:
        summary_points.append(f"{unresolved_count} conversations are flagged as unresolved.")
    if login_loops:
        summary_points.append(f"{login_loops} conversations show login-loop friction.")
    if order_friction:
        summary_points.append(f"{order_friction} conversations show order-flow friction.")
    if recommendation_conversations and recommendation_clicks == 0:
        summary_points.append("Recommendations are being shown, but they are not converting into clicks.")
    elif recommendation_conversations:
        summary_points.append(
            f"{recommendation_clicks} recommendation conversations led to user follow-through events."
        )

    return WidgetInsightSummary(
        widgetId=widget_id,
        brandName=brand_name,
        totalConversations=len(records),
        qualityBreakdown=dict(quality_breakdown),
        outcomeBreakdown=dict(outcome_breakdown),
        topIntents=[{"intent": intent, "count": count} for intent, count in intent_breakdown.most_common(5)],
        topProblems=[{"problem": problem, "count": count} for problem, count in problem_breakdown.most_common(5)],
        summaryPoints=summary_points,
    )
