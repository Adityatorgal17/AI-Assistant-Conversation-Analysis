from __future__ import annotations

from collections import Counter, defaultdict

from conversation_insights.insights_generator import (
    count_assistant_mistakes,
    count_user_mistakes,
    generate_global_insights,
    generate_widget_insights,
)
from conversation_insights.llm_insights_generator import LLMInsightEnhancer
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
                whoEndedConversation="unknown",
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
                qualityAccuracy=llm.qualityDimensions.accuracy,
                qualityRelevance=llm.qualityDimensions.relevance,
                qualityClarity=llm.qualityDimensions.clarity,
                qualityHelpfulness=llm.qualityDimensions.helpfulness,
                qualityTone=llm.qualityDimensions.tone,
                qualityEfficiency=llm.qualityDimensions.efficiency,
                qualityEscalationHandling=llm.qualityDimensions.escalation_handling,
                escalationNeeded=bool(llm.escalationNeeded),
                escalationTriggered=bool(llm.escalationTriggered),
                escalationResolved=bool(llm.escalationResolved),
            )
        )
    return rows


def build_widget_insights(records: list[ConversationFeatureRecord]) -> list[WidgetInsightSummary]:
    """Build widget insights with LLM enhancement (Path 2 only)."""
    widget_groups: dict[str, list[ConversationFeatureRecord]] = defaultdict(list)
    for record in records:
        widget_groups[record.widgetId].append(record)

    summaries: list[WidgetInsightSummary] = []
    enhancer = None

    # Initialize LLM enhancer
    try:
        enhancer = LLMInsightEnhancer()
        print("✓ Using LLM-powered insights (Path 2: base + LLM enhancement)")
    except RuntimeError as e:
        raise RuntimeError(f"LLM insights require API key: {e}")

    for widget_id, widget_records in sorted(widget_groups.items()):
        # Generate insights with LLM enhancement
        try:
            recommendations, assistant_mistakes, user_mistakes = enhancer.enhance_widget_insights(
                widget_id, widget_records
            )
        except Exception as e:
            print(f"⚠ LLM enhancement failed for {widget_id}: {e}")
            raise

        summaries.append(
            _build_widget_summary(widget_id, widget_records, recommendations, assistant_mistakes, user_mistakes)
        )

    return summaries


def build_global_summary(records: list[ConversationFeatureRecord]) -> GlobalInsightSummary:
    """Build global summary with LLM enhancement (Path 2 only)."""
    quality_breakdown = Counter(record.derivedMetrics.quality for record in records)
    outcome_breakdown = Counter((record.llmReview.conversationOutcome or "neutral") for record in records)
    top_intents = Counter((record.llmReview.initialIntent or "other") for record in records).most_common(5)
    top_problems = Counter(record.llmReview.primaryProblem for record in records if record.llmReview.primaryProblem).most_common(5)

    summary_points = [
        f"{quality_breakdown.get('bad', 0)} conversations are currently scored as bad quality.",
        f"The most common intent is {top_intents[0][0]}." if top_intents else "No intent summary available.",
        f"{outcome_breakdown.get('unresolved', 0)} conversations are flagged as unresolved.",
    ]

    # Generate widget-level insights
    widget_groups: dict[str, list[ConversationFeatureRecord]] = defaultdict(list)
    for record in records:
        widget_groups[record.widgetId].append(record)

    widget_insights = {}
    enhancer = None

    try:
        enhancer = LLMInsightEnhancer()
    except RuntimeError as e:
        raise RuntimeError(f"LLM insights require API key: {e}")

    for widget_id, widget_records in widget_groups.items():
        try:
            widget_insights[widget_id] = enhancer.enhance_widget_insights(widget_id, widget_records)
        except Exception as e:
            print(f"⚠ LLM enhancement failed for {widget_id}: {e}")
            raise

    # Generate global insights with LLM enhancement
    try:
        global_recommendations, cross_brand_findings = enhancer.enhance_global_insights(records, widget_insights)
    except Exception as e:
        print(f"⚠ Global LLM enhancement failed: {e}")
        raise

    # Count global mistakes
    assistant_mistakes_global = count_assistant_mistakes(records)
    user_mistakes_global = count_user_mistakes(records)
    quality_dimensions_global = _aggregate_quality_dimensions(records)

    return GlobalInsightSummary(
        totalConversations=len(records),
        qualityBreakdown=dict(quality_breakdown),
        outcomeBreakdown=dict(outcome_breakdown),
        topIntents=[{"intent": intent, "count": count} for intent, count in top_intents],
        topProblems=[{"problem": problem, "count": count} for problem, count in top_problems],
        summaryPoints=summary_points,
        recommendations=global_recommendations,
        crossBrandFindings=cross_brand_findings,
        assistantMistakesGlobal=assistant_mistakes_global,
        userMistakesGlobal=user_mistakes_global,
        qualityDimensionsGlobal=quality_dimensions_global,
    )


def _build_widget_summary(
    widget_id: str,
    records: list[ConversationFeatureRecord],
    recommendations: list = None,
    assistant_mistakes: dict = None,
    user_mistakes: dict = None,
) -> WidgetInsightSummary:
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

    quality_dimensions_aggregate = _aggregate_quality_dimensions(records)

    return WidgetInsightSummary(
        widgetId=widget_id,
        brandName=brand_name,
        totalConversations=len(records),
        qualityBreakdown=dict(quality_breakdown),
        outcomeBreakdown=dict(outcome_breakdown),
        topIntents=[{"intent": intent, "count": count} for intent, count in intent_breakdown.most_common(5)],
        topProblems=[{"problem": problem, "count": count} for problem, count in problem_breakdown.most_common(5)],
        summaryPoints=summary_points,
        recommendations=recommendations or [],
        assistantMistakes=assistant_mistakes or {},
        userMistakes=user_mistakes or {},
        qualityDimensionsAggregate=quality_dimensions_aggregate,
    )


def _aggregate_quality_dimensions(records: list[ConversationFeatureRecord]) -> dict[str, float]:
    if not records:
        return {
            "accuracy": 0.0,
            "relevance": 0.0,
            "clarity": 0.0,
            "helpfulness": 0.0,
            "tone": 0.0,
            "efficiency": 0.0,
            "escalation_handling": 0.0,
        }

    total = len(records)
    return {
        "accuracy": sum(r.llmReview.qualityDimensions.accuracy for r in records) / total,
        "relevance": sum(r.llmReview.qualityDimensions.relevance for r in records) / total,
        "clarity": sum(r.llmReview.qualityDimensions.clarity for r in records) / total,
        "helpfulness": sum(r.llmReview.qualityDimensions.helpfulness for r in records) / total,
        "tone": sum(r.llmReview.qualityDimensions.tone for r in records) / total,
        "efficiency": sum(r.llmReview.qualityDimensions.efficiency for r in records) / total,
        "escalation_handling": sum(r.llmReview.qualityDimensions.escalation_handling for r in records) / total,
    }

