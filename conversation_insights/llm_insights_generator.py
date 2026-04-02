"""LLM-powered insights generation: enhance base insights with LLM (Path 2 only).

This module generates widget and global insights using LLM enhancement:
- Generate base insights deterministically
- Send to LLM for root cause and action enhancement
- Return LLM-enriched insights as final output
"""

from __future__ import annotations

from collections import Counter
import json
import os
import re
import time
from typing import Any

from conversation_insights.config import PACKAGE_DIR
from conversation_insights.env_utils import load_env_file
from conversation_insights.insights_generator import (
    generate_global_insights,
    generate_widget_insights,
)
from conversation_insights.llm_review import discover_groq_keys, is_rate_limit_error, parse_retry_after_seconds
from conversation_insights.models import ConversationFeatureRecord, InsightRecommendation

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None


MODEL = "llama-3.3-70b-versatile"
MIN_SECONDS_BETWEEN_CALLS = 1.0
MAX_DISCOVERED_WIDGET_INSIGHTS = 2
MAX_DISCOVERED_GLOBAL_INSIGHTS = 2


class LLMInsightEnhancer:
    """Enhances base insights with LLM-generated root causes and actions (Path 2 only)."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize with Groq API key (required)."""
        if Groq is None:
            raise RuntimeError(
                "groq is not installed. Install dependencies from "
                "`conversation_insights/requirements.txt` first."
            )

        api_keys: list[str]
        if api_key is None:
            # Merge .env and process env so key lookup works in both local and exported setups.
            env_values = {
                **load_env_file(PACKAGE_DIR / ".env"),
                **{k: v for k, v in os.environ.items() if k.startswith("GROQ_API_KEY")},
            }
            api_keys = discover_groq_keys(env_values)
        else:
            api_keys = [api_key]

        if not api_keys:
            raise RuntimeError("No Groq API key found. Set GROQ_API_KEY in .env or pass api_key parameter.")

        self.clients = [Groq(api_key=key) for key in api_keys]
        self.client_index = 0
        self.client_available_at = [0.0 for _ in self.clients]
        self.client_last_call_at = [0.0 for _ in self.clients]

    def enhance_widget_insights(
        self,
        widget_id: str,
        widget_records: list[ConversationFeatureRecord],
    ) -> tuple[list[InsightRecommendation], dict[str, int], dict[str, int]]:
        """Generate widget insights with LLM enhancement (pure Path 2).

        Args:
            widget_id: Widget identifier
            widget_records: Conversations for this widget

        Returns:
            Tuple of (llm_enhanced_insights, assistant_mistakes, user_mistakes)
        """
        if not widget_records:
            return [], {}, {}

        # Generate base insights
        base_insights, assistant_mistakes, user_mistakes = generate_widget_insights(widget_records)

        if not base_insights:
            return base_insights, assistant_mistakes, user_mistakes

        # Enhance ALL insights with LLM (no selective filtering)
        enhanced_insights = []
        for insight in base_insights:
            try:
                enhanced = self._enhance_insight_with_llm(
                    widget_id=widget_id,
                    insight=insight,
                    widget_records=widget_records,
                )
                enhanced_insights.append(enhanced)
            except Exception as e:
                print(f"⚠ LLM enhancement failed for {insight.title}: {e}. Using base insight.")
                enhanced_insights.append(insight)

        try:
            discovered = self._discover_additional_widget_insights(
                widget_id=widget_id,
                widget_records=widget_records,
                existing_insights=enhanced_insights,
                assistant_mistakes=assistant_mistakes,
                user_mistakes=user_mistakes,
            )
            enhanced_insights.extend(discovered)
        except Exception as e:
            print(f"⚠ LLM discovery pass failed for widget {widget_id}: {e}. Continuing without discovered insights.")

        return enhanced_insights, assistant_mistakes, user_mistakes

    def enhance_global_insights(
        self,
        all_records: list[ConversationFeatureRecord],
        widget_insights: dict[str, tuple[list[InsightRecommendation], dict[str, int], dict[str, int]]],
    ) -> tuple[list[InsightRecommendation], dict[str, Any]]:
        """Generate global insights with LLM enhancement (pure Path 2).

        Args:
            all_records: All conversations across all widgets
            widget_insights: Insights per widget

        Returns:
            Tuple of (llm_enhanced_global_insights, cross_brand_findings)
        """
        base_insights, cross_brand_findings = generate_global_insights(all_records, widget_insights)

        if not base_insights:
            return base_insights, cross_brand_findings

        # Enhance ALL insights with LLM
        enhanced_insights = []
        for insight in base_insights:
            try:
                enhanced = self._enhance_global_insight_with_llm(
                    insight=insight,
                    cross_brand_findings=cross_brand_findings,
                    all_records=all_records,
                )
                enhanced_insights.append(enhanced)
            except Exception as e:
                print(f"⚠ LLM global enhancement failed for {insight.title}: {e}. Using base insight.")
                enhanced_insights.append(insight)

        try:
            discovered = self._discover_additional_global_insights(
                all_records=all_records,
                cross_brand_findings=cross_brand_findings,
                existing_insights=enhanced_insights,
            )
            enhanced_insights.extend(discovered)
        except Exception as e:
            print(f"⚠ LLM global discovery pass failed: {e}. Continuing without discovered insights.")

        return enhanced_insights, cross_brand_findings

    def _enhance_insight_with_llm(
        self,
        widget_id: str,
        insight: InsightRecommendation,
        widget_records: list[ConversationFeatureRecord],
    ) -> InsightRecommendation:
        """Use LLM to generate deeper root causes and actions for a single insight."""
        failed_conversations = [
            r for r in widget_records if self._insight_applies_to_record(insight, r)
        ]

        avg_quality_dims = self._extract_quality_patterns(failed_conversations)

        prompt = f"""You are an expert at analyzing customer service conversations to identify root causes and interventions.

WIDGET: {widget_id}
ISSUE: {insight.title}
AFFECTED: {insight.relatedCount}/{len(widget_records)} conversations ({insight.percentage:.1f}%)

Current Root Causes (reference):
{json.dumps(insight.rootCauses, indent=2)}

Current Suggested Actions (reference):
{json.dumps(insight.suggestedActions, indent=2)}

Quality Pattern in Affected Conversations:
- Average Accuracy: {avg_quality_dims.get('accuracy', 0):.1f}
- Average Relevance: {avg_quality_dims.get('relevance', 0):.1f}
- Average Clarity: {avg_quality_dims.get('clarity', 0):.1f}
- Average Helpfulness: {avg_quality_dims.get('helpfulness', 0):.1f}

Task: Provide enhanced root causes and actions based on the issue and quality patterns.
1. **Enhanced Root Causes** (2-3 specific, nuanced causes)
2. **Concrete Actions** (2-3 specific, testable actions prioritized by impact)

Return ONLY valid JSON in this exact format:
{{
  "root_causes": ["cause1", "cause2", "cause3"],
  "suggested_actions": ["action1", "action2", "action3"]
}}"""

        response = self._request_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at diagnosing customer service issues. Return ONLY valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        try:
            content = response.choices[0].message.content or ""
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_match:
                raise ValueError(f"No JSON found in response: {content}")

            enrichment = json.loads(json_match.group())
            return InsightRecommendation(
                title=insight.title,
                severity=insight.severity,
                relatedCount=insight.relatedCount,
                percentage=insight.percentage,
                description=insight.description,
                rootCauses=enrichment.get("root_causes", insight.rootCauses),
                suggestedActions=enrichment.get("suggested_actions", insight.suggestedActions),
                affectedCategories=insight.affectedCategories,
                metricBaseline=insight.metricBaseline,
                metricTarget=insight.metricTarget,
                interventionEffort=insight.interventionEffort,
                interventionRisk=insight.interventionRisk,
                hypothesis=insight.hypothesis,
                source="llm_enhanced",
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            raise ValueError(f"Failed to parse enrichment response: {e}")

    def _enhance_global_insight_with_llm(
        self,
        insight: InsightRecommendation,
        cross_brand_findings: dict[str, Any],
        all_records: list[ConversationFeatureRecord],
    ) -> InsightRecommendation:
        """Use LLM to generate strategic recommendations for global insights."""
        brand_comparison = cross_brand_findings.get("brandComparison", {})

        brand_stats = sorted(
            brand_comparison.items(),
            key=lambda x: x[1].get("badPercentage", 100),
        )
        best_performer = brand_stats[0][1] if brand_stats else {}
        worst_performer = brand_stats[-1][1] if brand_stats else {}

        prompt = f"""You are an analytics expert advising on cross-widget customer service improvements.

GLOBAL ISSUE: {insight.title}
AFFECTED: {insight.relatedCount}/{insight.percentage:.1f}% of conversations

RECOMMENDATION PRIORITIZATION:
- Effort: {insight.interventionEffort}
- Risk: {insight.interventionRisk}
- Baseline: {insight.metricBaseline}
- Target: {insight.metricTarget}

BRAND COMPARISON (Best vs Worst):
- Best Performer: {best_performer.get('brandName', 'N/A')} ({best_performer.get('badPercentage', 0):.1f}%)
- Worst Performer: {worst_performer.get('brandName', 'N/A')} ({worst_performer.get('badPercentage', 0):.1f}%)

CURRENT ACTIONS (reference):
{json.dumps(insight.suggestedActions, indent=2)}

Task: Provide 2-3 org-wide recommendations that account for best-practice insights from top performers and risk-appropriate implementation.

Return ONLY valid JSON:
{{
  "strategic_actions": ["action1", "action2", "action3"],
  "success_metrics": ["metric1", "metric2"]
}}"""

        response = self._request_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a strategic advisor. Return ONLY valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        try:
            content = response.choices[0].message.content or ""
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_match:
                raise ValueError(f"No JSON found in response: {content}")

            enrichment = json.loads(json_match.group())
            return InsightRecommendation(
                title=insight.title,
                severity=insight.severity,
                relatedCount=insight.relatedCount,
                percentage=insight.percentage,
                description=insight.description,
                rootCauses=insight.rootCauses,
                suggestedActions=enrichment.get("strategic_actions", insight.suggestedActions),
                affectedCategories=insight.affectedCategories,
                metricBaseline=insight.metricBaseline,
                metricTarget=insight.metricTarget,
                interventionEffort=insight.interventionEffort,
                interventionRisk=insight.interventionRisk,
                hypothesis=insight.hypothesis,
                source="llm_enhanced",
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            raise ValueError(f"Failed to parse global enrichment response: {e}")

    def _discover_additional_widget_insights(
        self,
        widget_id: str,
        widget_records: list[ConversationFeatureRecord],
        existing_insights: list[InsightRecommendation],
        assistant_mistakes: dict[str, int],
        user_mistakes: dict[str, int],
    ) -> list[InsightRecommendation]:
        if not widget_records:
            return []

        context = self._build_widget_discovery_context(
            widget_id=widget_id,
            widget_records=widget_records,
            existing_insights=existing_insights,
            assistant_mistakes=assistant_mistakes,
            user_mistakes=user_mistakes,
        )

        prompt = f"""You are a senior conversation analytics expert.

Task: discover up to {MAX_DISCOVERED_WIDGET_INSIGHTS} NEW issues that are not already covered by existing insights.

Guardrails:
- Only return issues supported by provided metrics.
- Do NOT repeat or rephrase existing insight titles.
- Prioritize minute-but-meaningful issues and major missed issues.
- Each issue must include concrete evidence metrics.
- If no strong unseen issue exists, return an empty list.
- Return ONLY valid JSON.

Input context (widget-level):
{json.dumps(context, ensure_ascii=True, indent=2)}

Output schema:
{{
  "discovered_insights": [
    {{
      "title": "string",
      "severity": "low|medium|high|critical",
      "description": "string",
      "root_causes": ["string", "string"],
      "suggested_actions": ["string", "string"],
      "affected_categories": ["string"],
      "confidence": 0.0,
      "why_new": "string",
      "evidence_metrics": {{"metric_name": "value"}}
    }}
  ]
}}"""

        response = self._request_completion(
            messages=[
                {"role": "system", "content": "Return only strict JSON matching schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        parsed = self._parse_json_response(response.choices[0].message.content or "")
        discovered = parsed.get("discovered_insights", [])
        if not isinstance(discovered, list):
            return []

        total = len(widget_records)
        normalized: list[InsightRecommendation] = []

        for item in discovered[:MAX_DISCOVERED_WIDGET_INSIGHTS]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue

            description = str(item.get("description", "")).strip()
            severity = str(item.get("severity", "medium")).lower()

            confidence = self._to_float_or_none(item.get("confidence"))
            if confidence is not None and confidence < 0.55:
                continue

            evidence = item.get("evidence_metrics", {})
            if not isinstance(evidence, dict):
                evidence = {}

            if self._is_duplicate_discovered_insight(
                title=title,
                description=description,
                evidence=evidence,
                existing_insights=existing_insights,
            ):
                continue

            coverage = self._derive_traceable_coverage(evidence=evidence, total=total)
            if coverage is None:
                # Skip discovered insights that cannot be tied back to concrete metrics.
                continue
            related_count, percentage, metric_baseline, metric_target = coverage

            insight = InsightRecommendation(
                title=title,
                severity=severity,
                relatedCount=related_count,
                percentage=percentage,
                description=description or "LLM-discovered issue from contextual metrics.",
                rootCauses=[str(x) for x in item.get("root_causes", []) if str(x).strip()][:3],
                suggestedActions=[str(x) for x in item.get("suggested_actions", []) if str(x).strip()][:3],
                affectedCategories=[str(x) for x in item.get("affected_categories", ["all"]) if str(x).strip()] or ["all"],
                metricBaseline=metric_baseline,
                metricTarget=metric_target,
                interventionEffort=self._default_effort_for_severity(severity),
                interventionRisk=self._default_risk_for_severity(severity),
                source="llm_discovery",
                confidence=confidence,
                whyNew=str(item.get("why_new", "")) or "Pattern was not covered by deterministic thresholds.",
                evidenceMetrics={k: v for k, v in evidence.items() if isinstance(k, str)},
            )

            # GUARDRAIL: Validate insight cleanliness before emission
            if not self._validate_insight_cleanliness(insight):
                continue

            normalized.append(insight)

        return normalized

    def _discover_additional_global_insights(
        self,
        all_records: list[ConversationFeatureRecord],
        cross_brand_findings: dict[str, Any],
        existing_insights: list[InsightRecommendation],
    ) -> list[InsightRecommendation]:
        if not all_records:
            return []

        context = self._build_global_discovery_context(
            all_records=all_records,
            cross_brand_findings=cross_brand_findings,
            existing_insights=existing_insights,
        )

        prompt = f"""You are a principal analytics strategist.

Task: discover up to {MAX_DISCOVERED_GLOBAL_INSIGHTS} NEW cross-widget issues not already covered.

Guardrails:
- Must be supported by provided data.
- Must not duplicate existing issues.
- Focus on meaningful patterns with operational impact.
- Include evidence metrics and confidence.
- Return empty list when no robust new issue exists.
- Return ONLY valid JSON.

Input context (global-level):
{json.dumps(context, ensure_ascii=True, indent=2)}

Output schema:
{{
  "discovered_insights": [
    {{
      "title": "string",
      "severity": "low|medium|high|critical",
      "description": "string",
      "root_causes": ["string", "string"],
      "suggested_actions": ["string", "string"],
      "affected_categories": ["string"],
      "confidence": 0.0,
      "why_new": "string",
      "evidence_metrics": {{"metric_name": "value"}}
    }}
  ]
}}"""

        response = self._request_completion(
            messages=[
                {"role": "system", "content": "Return only strict JSON matching schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        parsed = self._parse_json_response(response.choices[0].message.content or "")
        discovered = parsed.get("discovered_insights", [])
        if not isinstance(discovered, list):
            return []

        total = len(all_records)
        normalized: list[InsightRecommendation] = []

        for item in discovered[:MAX_DISCOVERED_GLOBAL_INSIGHTS]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue

            description = str(item.get("description", "")).strip()
            severity = str(item.get("severity", "medium")).lower()

            confidence = self._to_float_or_none(item.get("confidence"))
            if confidence is not None and confidence < 0.55:
                continue

            evidence = item.get("evidence_metrics", {})
            if not isinstance(evidence, dict):
                evidence = {}

            if self._is_duplicate_discovered_insight(
                title=title,
                description=description,
                evidence=evidence,
                existing_insights=existing_insights,
            ):
                continue

            coverage = self._derive_traceable_coverage(evidence=evidence, total=total)
            if coverage is None:
                continue
            related_count, percentage, metric_baseline, metric_target = coverage

            insight = InsightRecommendation(
                title=title,
                severity=severity,
                relatedCount=related_count,
                percentage=percentage,
                description=description or "LLM-discovered global issue from contextual metrics.",
                rootCauses=[str(x) for x in item.get("root_causes", []) if str(x).strip()][:3],
                suggestedActions=[str(x) for x in item.get("suggested_actions", []) if str(x).strip()][:3],
                affectedCategories=[str(x) for x in item.get("affected_categories", ["all"]) if str(x).strip()] or ["all"],
                metricBaseline=metric_baseline,
                metricTarget=metric_target,
                interventionEffort=self._default_effort_for_severity(severity),
                interventionRisk=self._default_risk_for_severity(severity),
                source="llm_discovery",
                confidence=confidence,
                whyNew=str(item.get("why_new", "")) or "Pattern was not captured by deterministic global rules.",
                evidenceMetrics={k: v for k, v in evidence.items() if isinstance(k, str)},
            )

            # GUARDRAIL: Validate insight cleanliness before emission
            if not self._validate_insight_cleanliness(insight):
                continue

            normalized.append(insight)

        return normalized

    def _build_widget_discovery_context(
        self,
        widget_id: str,
        widget_records: list[ConversationFeatureRecord],
        existing_insights: list[InsightRecommendation],
        assistant_mistakes: dict[str, int],
        user_mistakes: dict[str, int],
    ) -> dict[str, Any]:
        total = len(widget_records)
        intent_counts = Counter((r.llmReview.initialIntent or "other") for r in widget_records)
        outcome_counts = Counter((r.llmReview.conversationOutcome or "neutral") for r in widget_records)
        problem_counts = Counter(r.llmReview.primaryProblem for r in widget_records if r.llmReview.primaryProblem)
        quality_counts = Counter(r.derivedMetrics.quality for r in widget_records)

        drop_off_count = sum(bool(r.llmReview.dropOff) for r in widget_records)
        order_friction_count = sum(r.llmReview.primaryProblem == "order_friction" for r in widget_records)
        health_risk_count = sum(
            bool(r.llmReview.containsPossibleClaimRisk) and not bool(r.llmReview.containsSafetyDisclaimer)
            for r in widget_records
        )
        rec_given_count = sum(bool(r.llmReview.recommendationGiven) for r in widget_records)
        rec_converted_count = sum(
            bool(r.llmReview.recommendationGiven) and bool(r.llmReview.recommendationConverted)
            for r in widget_records
        )
        repetition_count = sum(bool(r.llmReview.assistantRepetition) for r in widget_records)
        login_loop_count = sum((r.llmReview.conversationOutcome or "") == "login_loop" for r in widget_records)

        rates = {
            "drop_off_rate": drop_off_count / total if total else 0.0,
            "order_friction_rate": order_friction_count / total if total else 0.0,
            "health_risk_rate": health_risk_count / total if total else 0.0,
            "recommendation_gap_rate": 1.0 - (rec_converted_count / rec_given_count) if rec_given_count else 0.0,
            "repetition_rate": repetition_count / total if total else 0.0,
            "login_loop_rate": login_loop_count / total if total else 0.0,
            "unresolved_rate": outcome_counts.get("unresolved", 0) / total if total else 0.0,
            "bad_quality_rate": quality_counts.get("bad", 0) / total if total else 0.0,
            "escalation_needed_rate": sum(bool(r.llmReview.escalationNeeded) for r in widget_records) / total if total else 0.0,
            "escalation_triggered_rate": sum(bool(r.llmReview.escalationTriggered) for r in widget_records) / total if total else 0.0,
        }

        summaries = [
            (r.llmReview.summary or "")
            for r in widget_records
            if r.llmReview.summary
        ][:8]

        return {
            "widgetId": widget_id,
            "brandName": widget_records[0].brandName if widget_records else None,
            "totalConversations": total,
            "knownInsights": [ins.title for ins in existing_insights],
            "qualityBreakdown": dict(quality_counts),
            "outcomeBreakdown": dict(outcome_counts),
            "intentBreakdown": dict(intent_counts),
            "problemBreakdown": dict(problem_counts),
            "rates": rates,
            "assistantMistakes": assistant_mistakes,
            "userMistakes": user_mistakes,
            "qualityDimensionsAggregate": self._extract_quality_patterns(widget_records),
            "interactionSignals": {
                "linkClickCountTotal": sum(r.conversationMeta.linkClickCount for r in widget_records),
                "productClickCountTotal": sum(r.conversationMeta.productClickCount for r in widget_records),
                "feedbackClickCountTotal": sum(r.conversationMeta.feedbackClickCount for r in widget_records),
                "productViewCountTotal": sum(r.conversationMeta.productViewCount for r in widget_records),
                "loginClickCountTotal": sum(r.conversationMeta.loginClickCount for r in widget_records),
            },
            "sampleSummaries": summaries,
        }

    def _build_global_discovery_context(
        self,
        all_records: list[ConversationFeatureRecord],
        cross_brand_findings: dict[str, Any],
        existing_insights: list[InsightRecommendation],
    ) -> dict[str, Any]:
        total = len(all_records)
        intent_counts = Counter((r.llmReview.initialIntent or "other") for r in all_records)
        outcome_counts = Counter((r.llmReview.conversationOutcome or "neutral") for r in all_records)
        problem_counts = Counter(r.llmReview.primaryProblem for r in all_records if r.llmReview.primaryProblem)
        quality_counts = Counter(r.derivedMetrics.quality for r in all_records)

        return {
            "totalConversations": total,
            "knownInsights": [ins.title for ins in existing_insights],
            "qualityBreakdown": dict(quality_counts),
            "outcomeBreakdown": dict(outcome_counts),
            "intentBreakdown": dict(intent_counts),
            "problemBreakdown": dict(problem_counts),
            "qualityDimensionsGlobal": self._extract_quality_patterns(all_records),
            "crossBrandFindings": cross_brand_findings,
            "assistantMistakesGlobal": {
                "hallucinating": sum(bool(r.llmReview.assistantHallucinating) for r in all_records),
                "wrong_product_suggestion": sum(bool(r.llmReview.assistantWrongProductSuggestion) for r in all_records),
                "health_claim_without_disclaimer": sum(bool(r.llmReview.assistantHealthClaimWithoutDisclaimer) for r in all_records),
                "failed_escalation": sum(bool(r.llmReview.assistantFailedEscalation) for r in all_records),
                "no_recommendation_when_needed": sum(bool(r.llmReview.assistantNoRecommendationWhenNeeded) for r in all_records),
            },
        }

    @staticmethod
    def _parse_json_response(content: str) -> dict[str, Any]:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {content}")
        parsed = json.loads(json_match.group())
        if not isinstance(parsed, dict):
            raise ValueError("Parsed response is not an object")
        return parsed

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _derive_traceable_coverage(evidence: dict[str, Any], total: int) -> tuple[int, float, float, float] | None:
        if total <= 0:
            return None

        # Count-like metrics can be used directly as affected volume.
        count_keys = (
            "affected_count",
            "count",
            "related_count",
            "risk_flagged_conversations",
            "unresolved_need",
            "health_claim_without_disclaimer",
            "did_not_provide_required_info",
            "order_friction_count",
            "drop_off_count",
            "login_loop_count",
            "repetition_count",
        )
        count_value: float | None = None
        for key in count_keys:
            metric = LLMInsightEnhancer._to_float_or_none(evidence.get(key))
            if metric is not None and metric > 0:
                count_value = metric
                break

        # Rate-like metrics can be converted into affected volume.
        rate_keys = (
            "affected_rate",
            "rate",
            # Prefer issue-specific rates before generic outcome rates.
            "order_friction_rate",
            "drop_off_rate",
            "health_risk_rate",
            "recommendation_gap_rate",
            "repetition_rate",
            "login_loop_rate",
            "escalation_needed_rate",
            "escalation_triggered_rate",
            "unresolved_rate",
            "bad_quality_rate",
        )
        percent_keys = (
            "affected_percent",
            "rate_percent",
            "percentage",
        )

        rate_value: float | None = None
        for key in rate_keys:
            metric = LLMInsightEnhancer._to_float_or_none(evidence.get(key))
            if metric is not None and metric > 0:
                rate_value = metric
                break

        if rate_value is None:
            for key in percent_keys:
                metric = LLMInsightEnhancer._to_float_or_none(evidence.get(key))
                if metric is not None and metric > 0:
                    rate_value = metric / 100.0
                    break

        if count_value is None and rate_value is None:
            return None

        if count_value is None and rate_value is not None:
            count_value = rate_value * total
        if rate_value is None and count_value is not None:
            rate_value = count_value / total

        if count_value is None or rate_value is None:
            return None

        # GUARDRAIL 3: Validate cleanliness - reject relatedCount if it appears inflated or conflated
        # If derived count is > 1.2x more than total, something is wrong
        if count_value > total * 1.05:
            return None

        # Reject if percentage is 0.0 or suspiciously precise (like 95.3125, suggesting raw decimal masquerading as percentage)
        suspicious_percentages = [0.0, rate_value * 100 if rate_value * 100 > 10 and rate_value * 100 % 1 != 0 else None]
        if rate_value * 100 in suspicious_percentages:
            return None

        related_count = max(1, min(total, int(round(count_value))))
        baseline = max(0.0, min(1.0, float(rate_value)))
        percentage = round(baseline * 100, 2)

        # Default target aims for measurable reduction without overpromising.
        target = round(max(0.0, baseline - 0.05), 4)

        return related_count, percentage, round(baseline, 4), target

    @staticmethod
    def _default_effort_for_severity(severity: str) -> str:
        return {
            "critical": "high",
            "high": "medium",
            "medium": "medium",
            "low": "low",
        }.get((severity or "medium").lower(), "medium")

    @staticmethod
    def _default_risk_for_severity(severity: str) -> str:
        return {
            "critical": "high",
            "high": "medium",
            "medium": "low",
            "low": "low",
        }.get((severity or "medium").lower(), "low")

    @staticmethod
    def _normalize_title_tokens(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", (value or "").lower())
            if len(token) > 2
        }

    @staticmethod
    def _validate_insight_cleanliness(insight: InsightRecommendation) -> bool:
        """GUARDRAIL: Validate insight passes all quality checks before emission.
        
        Checks:
        1. relatedCount must be traceable (not 0, not > total)
        2. baseline/target/percentage must be cleanly populated (not None, not 0.0 placeholders)
        3. confidence should be set if insight is discovery
        """
        # Check 1: relatedCount must be reasonable
        if insight.relatedCount is None or insight.relatedCount <= 0:
            return False

        # Check 2: percentage must be properly populated (not 0.0 or other placeholder)
        if insight.percentage is None or insight.percentage == 0.0:
            return False

        # Check 3: if source is llm_discovery, confidence should be set
        if insight.source == "llm_discovery":
            if insight.confidence is None or insight.confidence < 0.55:
                return False

        # Check 4: baseline/target should be either both populated or both absent
        if insight.source == "llm_discovery":
            has_baseline = insight.metricBaseline is not None and insight.metricBaseline > 0
            has_target = insight.metricTarget is not None
            if has_baseline != has_target:
                # One is set but not the other - inconsistent
                return False

        return True

    @staticmethod
    def _extract_topic_tags(text: str) -> set[str]:
        normalized = (text or "").lower()
        topic_keywords = {
            "compliance": ("risk", "compliance", "safety", "claim", "disclaimer", "flag"),
            "escalation": ("escalation", "handoff", "hand-off", "transfer"),
            "unresolved": ("unresolved", "unresolv", "not resolved", "resolution"),
            "dropoff": ("drop-off", "drop off", "dropoff", "abandon"),
            "recommendation": ("recommend", "conversion", "follow-through", "follow through"),
            "order_friction": ("order", "friction", "tracking"),
            "login": ("login", "auth", "authentication"),
            "repetition": ("repetition", "repeat"),
            "quality": ("accuracy", "relevance", "clarity", "helpfulness", "efficiency", "tone", "quality"),
        }
        tags: set[str] = set()
        for tag, keywords in topic_keywords.items():
            if any(keyword in normalized for keyword in keywords):
                tags.add(tag)
        return tags

    def _is_duplicate_discovered_insight(
        self,
        title: str,
        description: str,
        evidence: dict[str, Any],
        existing_insights: list[InsightRecommendation],
    ) -> bool:
        title_norm = title.strip().lower()
        if not title_norm:
            return True

        candidate_tokens = self._normalize_title_tokens(title_norm)
        evidence_text = " ".join(str(key) for key in evidence.keys())
        candidate_topics = self._extract_topic_tags(f"{title} {description} {evidence_text}")

        # GUARDRAIL 1: Strict exact match on metrics (e.g., repetition_rate exact match indicates deterministic overlap)
        for key in evidence.keys():
            if any(det_key in str(key).lower() for det_key in ["repetition", "unresolved", "order_friction", "escalation", "health_risk"]):
                # If LLM discovery uses a deterministic raw metric key, it's likely a duplicate
                if str(key).lower() in ["repetition_count", "unresolved_count", "order_friction_count", "health_claim_without_disclaimer"]:
                    return True

        for existing in existing_insights:
            existing_title_norm = existing.title.strip().lower()
            if title_norm == existing_title_norm:
                return True

            existing_tokens = self._normalize_title_tokens(existing_title_norm)
            if candidate_tokens and existing_tokens:
                overlap = len(candidate_tokens & existing_tokens) / len(candidate_tokens | existing_tokens)
                if overlap >= 0.6:
                    return True

            existing_topics = self._extract_topic_tags(f"{existing.title} {existing.description}")
            if candidate_topics and existing_topics and candidate_topics & existing_topics:
                # GUARDRAIL 2: Stricter topic overlap (any single topic overlap is now a duplicate)
                return True

        return False

    def _request_completion(self, messages: list[dict[str, str]], temperature: float) -> Any:
        """Create a completion with automatic key rotation when rate limits are hit."""
        attempts_left = max(len(self.clients) * 6, 6)
        last_error: Exception | None = None

        while attempts_left > 0:
            self.client_index = self._wait_for_available_client()
            self._respect_rate_limit(self.client_index)
            client = self.clients[self.client_index]
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    temperature=temperature,
                    messages=messages,
                )
                if len(self.clients) > 1:
                    self.client_index = (self.client_index + 1) % len(self.clients)
                return response
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if is_rate_limit_error(exc):
                    wait_seconds = parse_retry_after_seconds(str(exc)) or 60.0
                    self.client_available_at[self.client_index] = time.monotonic() + wait_seconds + 2.0
                    if len(self.clients) > 1:
                        print(
                            f"Rate limit hit on insights client {self.client_index + 1}/{len(self.clients)}; "
                            f"cooling down for {wait_seconds:.1f}s and switching."
                        )
                        continue
                    print(
                        f"Rate limit hit on insights client {self.client_index + 1}/{len(self.clients)}; "
                        f"waiting {wait_seconds:.1f}s before retrying."
                    )
                    time.sleep(wait_seconds + 2.0)
                    continue
                attempts_left -= 1
                time.sleep(2)

        raise RuntimeError(f"LLM insight enhancement failed after retries: {last_error}")

    def _respect_rate_limit(self, client_index: int) -> None:
        """Enforce minimum time between API calls for a specific client."""
        elapsed = time.monotonic() - self.client_last_call_at[client_index]
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
        self.client_last_call_at[client_index] = time.monotonic()

    def _wait_for_available_client(self) -> int:
        now = time.monotonic()
        best_index = min(range(len(self.client_available_at)), key=lambda idx: self.client_available_at[idx])
        available_at = self.client_available_at[best_index]
        if available_at > now:
            time.sleep(available_at - now)
        return best_index

    @staticmethod
    def _insight_applies_to_record(
        insight: InsightRecommendation,
        record: ConversationFeatureRecord,
    ) -> bool:
        """Check if a record matches the insight criteria."""
        title_lower = insight.title.lower()

        if "drop-off" in title_lower:
            return bool(record.llmReview.dropOff)
        elif "order" in title_lower and "friction" in title_lower:
            return record.llmReview.primaryProblem == "order_friction"
        elif "compliance" in title_lower or "safety" in title_lower:
            return bool(record.llmReview.containsPossibleClaimRisk) and not bool(
                record.llmReview.containsSafetyDisclaimer
            )
        elif "recommendation" in title_lower and "follow" in title_lower:
            return bool(record.llmReview.recommendationGiven) and not bool(record.llmReview.recommendationConverted)
        elif "repetition" in title_lower:
            return bool(record.llmReview.assistantRepetition)
        elif "login" in title_lower or "auth" in title_lower:
            return record.llmReview.conversationOutcome == "login_loop"
        elif "unresolved" in title_lower:
            return bool(record.llmReview.unresolved)

        return False

    @staticmethod
    def _extract_quality_patterns(records: list[ConversationFeatureRecord]) -> dict[str, float]:
        """Extract average quality dimensions from records."""
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
