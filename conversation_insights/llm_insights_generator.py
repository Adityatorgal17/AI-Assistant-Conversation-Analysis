"""LLM-powered insights generation: enhance base insights with LLM (Path 2 only).

This module generates widget and global insights using LLM enhancement:
- Generate base insights deterministically
- Send to LLM for root cause and action enhancement
- Return LLM-enriched insights as final output
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
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
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            raise ValueError(f"Failed to parse global enrichment response: {e}")

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
