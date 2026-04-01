from __future__ import annotations

from typing import Any, Callable

from conversation_insights.models import ConversationFeatureRecord, InsightRecommendation


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _dynamic_trigger_threshold(rates: list[float], floor: float = 0.08, pct: float = 70.0) -> float:
    return max(floor, _percentile(rates, pct))


def _count(records: list[ConversationFeatureRecord], predicate: Callable[[ConversationFeatureRecord], bool]) -> int:
    return sum(1 for record in records if predicate(record))


def generate_widget_insights(
    records: list[ConversationFeatureRecord],
) -> tuple[list[InsightRecommendation], dict[str, int], dict[str, int]]:
    recommendations: list[InsightRecommendation] = []
    assistant_mistakes = count_assistant_mistakes(records)
    user_mistakes = count_user_mistakes(records)

    if not records:
        return recommendations, assistant_mistakes, user_mistakes

    total = len(records)
    drop_off_count = _count(records, lambda r: bool(r.llmReview.dropOff))
    order_friction_count = _count(records, lambda r: r.llmReview.primaryProblem == "order_friction")
    health_risk_count = _count(
        records,
        lambda r: bool(r.llmReview.containsPossibleClaimRisk) and not bool(r.llmReview.containsSafetyDisclaimer),
    )
    rec_given_count = _count(records, lambda r: bool(r.llmReview.recommendationGiven))
    rec_converted_count = _count(records, lambda r: bool(r.llmReview.recommendationGiven) and bool(r.llmReview.recommendationConverted))
    repetition_count = _count(records, lambda r: bool(r.llmReview.assistantRepetition))
    login_loop_count = _count(records, lambda r: r.llmReview.conversationOutcome == "login_loop")

    drop_off_rate = _safe_rate(drop_off_count, total)
    order_friction_rate = _safe_rate(order_friction_count, total)
    health_risk_rate = _safe_rate(health_risk_count, total)
    recommendation_gap_rate = 1.0 - _safe_rate(rec_converted_count, rec_given_count) if rec_given_count else 0.0
    repetition_rate = _safe_rate(repetition_count, total)
    login_loop_rate = _safe_rate(login_loop_count, total)

    rates = [
        drop_off_rate,
        order_friction_rate,
        health_risk_rate,
        recommendation_gap_rate,
        repetition_rate,
        login_loop_rate,
    ]
    trigger_threshold = _dynamic_trigger_threshold(rates, floor=0.08, pct=70.0)

    if drop_off_rate >= trigger_threshold and drop_off_count >= 3:
        recommendations.append(
            InsightRecommendation(
                title="Above-Baseline User Drop-Off",
                severity="high" if drop_off_rate >= 0.25 else "medium",
                relatedCount=drop_off_count,
                percentage=drop_off_rate * 100,
                description=f"{drop_off_rate * 100:.1f}% conversations dropped off, above this widget's dynamic risk threshold.",
                rootCauses=[
                    "Users are not receiving enough value in early turns",
                    "Assistant responses may be too generic for initial needs",
                    "Conversation entry points may be unclear",
                ],
                suggestedActions=[
                    "Improve first response templates for top intents",
                    "Add intent-specific quick actions in the first turn",
                    "A/B test concise vs detailed first responses",
                ],
                affectedCategories=["all"],
                metricBaseline=drop_off_rate,
                metricTarget=max(0.0, drop_off_rate - 0.08),
                interventionEffort="medium",
                interventionRisk="low",
                hypothesis="Reducing first-turn friction will improve continuation and lower drop-off.",
            )
        )

    if order_friction_rate >= trigger_threshold and order_friction_count >= 3:
        recommendations.append(
            InsightRecommendation(
                title="Order-Support Friction Cluster",
                severity="high",
                relatedCount=order_friction_count,
                percentage=order_friction_rate * 100,
                description=f"{order_friction_count} conversations stalled in order-support flows.",
                rootCauses=[
                    "High information burden for users in support flow",
                    "Limited fallback path when primary lookup fails",
                    "Insufficient progressive guidance",
                ],
                suggestedActions=[
                    "Split information requests into smaller guided steps",
                    "Offer fallback lookup options after first failure",
                    "Trigger escalation after repeated failed attempts",
                ],
                affectedCategories=["order_support"],
                metricBaseline=order_friction_rate,
                metricTarget=max(0.0, order_friction_rate - 0.07),
                interventionEffort="medium",
                interventionRisk="low",
                hypothesis="Progressive collection and fallback paths will reduce order-flow abandonment.",
            )
        )

    if health_risk_rate >= trigger_threshold and health_risk_count >= 3:
        recommendations.append(
            InsightRecommendation(
                title="Compliance Risk: Missing Safety Framing",
                severity="critical",
                relatedCount=health_risk_count,
                percentage=health_risk_rate * 100,
                description=f"{health_risk_count} conversations include risk-sensitive claims without disclaimers.",
                rootCauses=[
                    "Safety language is not consistently injected",
                    "Risk-sensitive intents are not isolated early",
                    "Claim moderation checks are insufficient",
                ],
                suggestedActions=[
                    "Inject standardized disclaimer templates for risk-sensitive contexts",
                    "Add a pre-response compliance check for sensitive claims",
                    "Route high-risk conversations to escalation workflow",
                ],
                affectedCategories=["risk_and_compliance"],
                metricBaseline=health_risk_rate,
                metricTarget=max(0.0, health_risk_rate - 0.15),
                interventionEffort="low",
                interventionRisk="medium",
                hypothesis="Automated safety framing will materially reduce compliance exposure.",
            )
        )

    if rec_given_count >= 5 and recommendation_gap_rate >= trigger_threshold:
        conversion_rate = _safe_rate(rec_converted_count, rec_given_count)
        recommendations.append(
            InsightRecommendation(
                title="Low Recommendation Follow-Through",
                severity="medium",
                relatedCount=rec_given_count - rec_converted_count,
                percentage=recommendation_gap_rate * 100,
                description=f"Recommendation conversion is {conversion_rate * 100:.1f}% for this widget.",
                rootCauses=[
                    "Recommendations are not consistently matched to intent",
                    "Call-to-action visibility may be weak",
                    "User confidence in recommendations is low",
                ],
                suggestedActions=[
                    "Prioritize high-confidence recommendations only",
                    "Add explanation snippets for why items were suggested",
                    "Test stronger CTA formatting for suggested products",
                ],
                affectedCategories=["product_discovery", "product_page_question"],
                metricBaseline=conversion_rate,
                metricTarget=min(1.0, conversion_rate + 0.10),
                interventionEffort="medium",
                interventionRisk="low",
                hypothesis="Higher recommendation precision and clearer CTAs will improve conversion.",
            )
        )

    if repetition_rate >= trigger_threshold and repetition_count >= 3:
        recommendations.append(
            InsightRecommendation(
                title="Assistant Repetition Pattern",
                severity="medium",
                relatedCount=repetition_count,
                percentage=repetition_rate * 100,
                description=f"{repetition_rate * 100:.1f}% conversations exhibit repetitive assistant behavior.",
                rootCauses=[
                    "Loop-breaking logic is weak",
                    "Escalation trigger conditions are too late",
                    "Clarification strategy is not adaptive",
                ],
                suggestedActions=[
                    "Add loop-break rule after repeated prompts",
                    "Escalate sooner when identical asks recur",
                    "Introduce alternative-response templates",
                ],
                affectedCategories=["all"],
                metricBaseline=repetition_rate,
                metricTarget=max(0.0, repetition_rate - 0.08),
                interventionEffort="low",
                interventionRisk="low",
                hypothesis="Loop detection with early alternatives will reduce repetition frequency.",
            )
        )

    if login_loop_rate >= trigger_threshold and login_loop_count >= 2:
        recommendations.append(
            InsightRecommendation(
                title="Authentication Loop Friction",
                severity="high",
                relatedCount=login_loop_count,
                percentage=login_loop_rate * 100,
                description=f"{login_loop_count} conversations were trapped in login loops.",
                rootCauses=[
                    "Authentication fallback path is unclear",
                    "Repeated sign-in prompts without resolution",
                    "Escalation is not triggered quickly enough",
                ],
                suggestedActions=[
                    "Offer alternate verification path after first failure",
                    "Suppress repeated sign-in prompts",
                    "Escalate to support after repeated auth failures",
                ],
                affectedCategories=["order_support"],
                metricBaseline=login_loop_rate,
                metricTarget=max(0.0, login_loop_rate - 0.08),
                interventionEffort="medium",
                interventionRisk="low",
                hypothesis="Reducing auth loops and adding fallback verification will improve resolution rates.",
            )
        )

    return recommendations, assistant_mistakes, user_mistakes


def generate_global_insights(
    all_records: list[ConversationFeatureRecord],
    widget_insights: dict[str, tuple[list[InsightRecommendation], dict[str, int], dict[str, int]]],
) -> tuple[list[InsightRecommendation], dict[str, Any]]:
    recommendations: list[InsightRecommendation] = []
    cross_brand_findings: dict[str, Any] = {}

    if not all_records:
        return recommendations, cross_brand_findings

    brand_quality: dict[str, dict[str, Any]] = {}
    for widget_id in widget_insights:
        widget_records = [record for record in all_records if record.widgetId == widget_id]
        if not widget_records:
            continue
        bad_count = _count(widget_records, lambda r: r.derivedMetrics.quality == "bad")
        bad_rate = _safe_rate(bad_count, len(widget_records))
        brand_quality[widget_id] = {
            "brandName": widget_records[0].brandName,
            "badPercentage": bad_rate * 100,
            "totalConversations": len(widget_records),
        }

    cross_brand_findings["brandComparison"] = brand_quality

    bad_rates = [entry["badPercentage"] / 100.0 for entry in brand_quality.values()]
    bad_rate_trigger = _dynamic_trigger_threshold(bad_rates, floor=0.45, pct=75.0)

    for widget_id, stats in brand_quality.items():
        bad_rate = stats["badPercentage"] / 100.0
        if bad_rate >= bad_rate_trigger and stats["totalConversations"] >= 20:
            recommendations.append(
                InsightRecommendation(
                    title=f"Brand {stats['brandName']} Above Bad-Quality Baseline",
                    severity="critical" if bad_rate >= 0.60 else "high",
                    relatedCount=round(stats["totalConversations"] * bad_rate),
                    percentage=stats["badPercentage"],
                    description=(
                        f"{stats['brandName']} has {stats['badPercentage']:.1f}% bad-quality conversations, "
                        "above dynamic cross-brand threshold."
                    ),
                    rootCauses=[
                        "Brand-specific intent mix may be harder",
                        "Knowledge coverage may be weaker for this brand",
                        "Escalation path may be less effective",
                    ],
                    suggestedActions=[
                        f"Run targeted failure-mode analysis for {stats['brandName']}",
                        "Compare top intents against best-performing brand",
                        "Prioritize fixes in highest-volume failure intents",
                    ],
                    affectedCategories=["all"],
                    metricBaseline=bad_rate,
                    metricTarget=max(0.0, bad_rate - 0.10),
                    interventionEffort="medium",
                    interventionRisk="medium",
                    hypothesis="Targeted brand-level fixes should materially reduce bad-quality rates.",
                )
            )

    all_health_risk = _count(
        all_records,
        lambda r: bool(r.llmReview.containsPossibleClaimRisk) and not bool(r.llmReview.containsSafetyDisclaimer),
    )
    health_risk_rate = _safe_rate(all_health_risk, len(all_records))

    unresolved_resolvable = _count(
        all_records,
        lambda r: bool(r.llmReview.unresolved) and bool(r.llmReview.problemCouldBeResolved),
    )
    unresolved_resolvable_rate = _safe_rate(unresolved_resolvable, len(all_records))

    global_rates = [health_risk_rate, unresolved_resolvable_rate]
    global_trigger = _dynamic_trigger_threshold(global_rates, floor=0.10, pct=70.0)

    if health_risk_rate >= global_trigger and all_health_risk >= 5:
        recommendations.append(
            InsightRecommendation(
                title="Global Compliance Gap",
                severity="critical",
                relatedCount=all_health_risk,
                percentage=health_risk_rate * 100,
                description=f"{all_health_risk} conversations include risk-sensitive claims without disclaimer coverage.",
                rootCauses=[
                    "Compliance checks are inconsistently applied",
                    "Risk-sensitive wording is not always detected",
                    "Fallback disclaimer insertion is missing",
                ],
                suggestedActions=[
                    "Apply centralized compliance guardrail before assistant response",
                    "Expand risk term detection coverage",
                    "Require disclaimer templates for flagged responses",
                ],
                affectedCategories=["risk_and_compliance"],
                metricBaseline=health_risk_rate,
                metricTarget=max(0.0, health_risk_rate - 0.12),
                interventionEffort="low",
                interventionRisk="medium",
                hypothesis="Centralized compliance checks should reduce safety-risk output incidence.",
            )
        )

    if unresolved_resolvable_rate >= global_trigger and unresolved_resolvable >= 5:
        recommendations.append(
            InsightRecommendation(
                title="Avoidable Unresolved Conversations",
                severity="high",
                relatedCount=unresolved_resolvable,
                percentage=unresolved_resolvable_rate * 100,
                description=(
                    f"{unresolved_resolvable} conversations were unresolved despite having a potentially resolvable path."
                ),
                rootCauses=[
                    "Escalation decisioning is delayed",
                    "Fallback alternatives are not consistently offered",
                    "Clarification strategy fails for complex intents",
                ],
                suggestedActions=[
                    "Add escalation trigger for repeated failed turns",
                    "Offer alternatives before ending unresolved",
                    "Route high-friction intents to stronger playbooks",
                ],
                affectedCategories=["all"],
                metricBaseline=unresolved_resolvable_rate,
                metricTarget=max(0.0, unresolved_resolvable_rate - 0.10),
                interventionEffort="medium",
                interventionRisk="low",
                hypothesis="Earlier escalation and fallback strategies will reduce avoidable unresolved outcomes.",
            )
        )

    cross_brand_findings["totalConversations"] = len(all_records)
    cross_brand_findings["globalBadQualityPercentage"] = _safe_rate(
        _count(all_records, lambda r: r.derivedMetrics.quality == "bad"),
        len(all_records),
    ) * 100

    return recommendations, cross_brand_findings


def count_assistant_mistakes(records: list[ConversationFeatureRecord]) -> dict[str, int]:
    return {
        "hallucinating": _count(records, lambda r: bool(r.llmReview.assistantHallucinating)),
        "wrong_product_suggestion": _count(records, lambda r: bool(r.llmReview.assistantWrongProductSuggestion)),
        "health_claim_without_disclaimer": _count(records, lambda r: bool(r.llmReview.assistantHealthClaimWithoutDisclaimer)),
        "failed_escalation": _count(records, lambda r: bool(r.llmReview.assistantFailedEscalation)),
        "no_recommendation_when_needed": _count(records, lambda r: bool(r.llmReview.assistantNoRecommendationWhenNeeded)),
        "repetition": _count(records, lambda r: bool(r.llmReview.assistantRepetition)),
        "short_answer": _count(records, lambda r: bool(r.llmReview.assistantShortAnswer)),
        "evasive_answer": _count(records, lambda r: bool(r.llmReview.assistantEvasiveAnswer)),
    }


def count_user_mistakes(records: list[ConversationFeatureRecord]) -> dict[str, int]:
    return {
        "never_responded": _count(records, lambda r: bool(r.llmReview.userNeverResponded)),
        "did_not_provide_required_info": _count(records, lambda r: bool(r.llmReview.userDidNotProvideRequiredInfo)),
        "asked_unrelated_questions": _count(records, lambda r: bool(r.llmReview.userAskedUnrelatedQuestions)),
        "ordering_mistake": _count(records, lambda r: bool(r.llmReview.userOrderingMistake)),
        "did_not_follow_instructions": _count(records, lambda r: bool(r.llmReview.userDidNotFollowInstructions)),
    }
