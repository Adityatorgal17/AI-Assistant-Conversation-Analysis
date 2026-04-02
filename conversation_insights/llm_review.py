from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from conversation_insights.config import PACKAGE_DIR
from conversation_insights.env_utils import load_env_file
from conversation_insights.models import ConversationFeatureRecord, GroupedConversationRecord, QualityDimensions
from conversation_insights.text_utils import (
    ORDER_KEYWORDS,
    PRODUCT_DISCOVERY_KEYWORDS,
    PRODUCT_PAGE_PREFIXES,
    detect_language_style,
    has_slug_like_token,
)

MODEL = "llama-3.3-70b-versatile"
MIN_SECONDS_BETWEEN_CALLS = 2.0
CACHE_SCHEMA_VERSION = 1

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None


@dataclass(slots=True)
class ReviewResult:
    initial_intent: str
    language_style: str
    has_safety_sensitive_context: bool
    user_repetition: bool
    frustrated: bool
    assistant_repetition: bool
    assistant_short_answer: bool
    assistant_evasive_answer: bool
    contains_order_instructions: bool
    contains_safety_disclaimer: bool
    contains_possible_claim_risk: bool
    recommendation_given: bool
    recommendation_relevant: bool
    bad_recommendation: bool
    success: bool
    drop_off: bool
    unresolved: bool
    primary_problem: str | None
    conversation_outcome: str
    issues: list[str]
    summary: str
    feedback_text: str
    user_never_responded: bool = False
    user_did_not_provide_required_info: bool = False
    user_asked_unrelated_questions: bool = False
    user_ordering_mistake: bool = False
    user_did_not_follow_instructions: bool = False
    assistant_hallucinating: bool = False
    assistant_wrong_product_suggestion: bool = False
    assistant_health_claim_without_disclaimer: bool = False
    assistant_failed_escalation: bool = False
    assistant_no_recommendation_when_needed: bool = False
    user_engaged: bool = False
    recommendation_converted: bool = False
    problem_could_be_resolved: bool = False

    # New production fields
    quality_dimensions: QualityDimensions = field(default_factory=QualityDimensions)
    escalation_needed: bool | None = None
    escalation_triggered: bool | None = None
    escalation_resolved: bool | None = None
    time_to_escalation_seconds: int | None = None
    resolution_blockers: dict[str, str] = field(default_factory=dict)


class GroqReviewer:
    def __init__(self, api_keys: list[str], cache_path: Path) -> None:
        if Groq is None:
            raise RuntimeError(
                "groq is not installed. Install dependencies from "
                "`conversation_insights/requirements.txt` first."
            )
        if not api_keys:
            raise RuntimeError("No Groq API key was found in conversation_insights/.env")

        self.clients = [Groq(api_key=key) for key in api_keys]
        self.client_index = 0
        self.client_available_at = [0.0 for _ in self.clients]
        self.client_last_call_at = [0.0 for _ in self.clients]
        self.cache_path = cache_path
        self.cache = self._load_cache()

    def review_conversations(
        self,
        grouped_records: list[GroupedConversationRecord],
        feature_records: list[ConversationFeatureRecord],
    ) -> list[ConversationFeatureRecord]:
        by_conversation = {record.conversationId: record for record in grouped_records}
        reviewed_records: list[ConversationFeatureRecord] = []

        for index, feature_record in enumerate(feature_records, start=1):
            cached = self.cache.get(feature_record.conversationId)
            grouped_record = by_conversation[feature_record.conversationId]
            if cached and cached.get("__schemaVersion") == CACHE_SCHEMA_VERSION:
                result = parse_review_data(cached)
            else:
                prompt = build_prompt(grouped_record, feature_record)
                result = self._request_review(prompt)
                cache_entry = asdict(result)
                cache_entry["__schemaVersion"] = CACHE_SCHEMA_VERSION
                self.cache[feature_record.conversationId] = cache_entry
                self._write_cache()

            result = finalize_review_result(grouped_record, feature_record, result)
            apply_review_result(feature_record, result)
            reviewed_records.append(feature_record)

            if index % 25 == 0:
                print(f"LLM reviewed {index}/{len(feature_records)} conversations")

        return reviewed_records

    def _request_review(self, prompt: str) -> ReviewResult:
        attempts_left = max(len(self.clients) * 6, 6)
        last_error: Exception | None = None

        while attempts_left > 0:
            self.client_index = self._wait_for_available_client()
            self._respect_rate_limit(self.client_index)
            client = self.clients[self.client_index]
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You review one assistant conversation with high precision. "
                                "Return ONLY valid JSON. Do not add markdown, prose, or code fences."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                content = response.choices[0].message.content or ""
                if len(self.clients) > 1:
                    self.client_index = (self.client_index + 1) % len(self.clients)
                return parse_review_json(content)
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if is_rate_limit_error(exc):
                    wait_seconds = parse_retry_after_seconds(str(exc)) or 60.0
                    self.client_available_at[self.client_index] = time.monotonic() + wait_seconds + 2.0
                    if len(self.clients) > 1:
                        print(
                            f"Rate limit hit on client {self.client_index + 1}/{len(self.clients)}; "
                            f"cooling down for {wait_seconds:.1f}s and switching."
                        )
                        continue
                    print(
                        f"Rate limit hit on client {self.client_index + 1}/{len(self.clients)}; "
                        f"waiting {wait_seconds:.1f}s before retrying."
                    )
                    time.sleep(wait_seconds + 2.0)
                    continue
                attempts_left -= 1
                time.sleep(2)

        raise RuntimeError(f"LLM review failed after retries: {last_error}")

    def _respect_rate_limit(self, client_index: int) -> None:
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

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        return json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _write_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2, ensure_ascii=True), encoding="utf-8")


def run_llm_reviews(
    grouped_records: list[GroupedConversationRecord],
    feature_records: list[ConversationFeatureRecord],
    output_dir: Path,
) -> list[ConversationFeatureRecord]:
    env_values = load_env_file(PACKAGE_DIR / ".env")
    api_keys = discover_groq_keys(env_values)
    if not api_keys:
        raise RuntimeError(
            "No Groq API key found in conversation_insights/.env. "
            "LLM review is mandatory for all conversations."
        )

    reviewer = GroqReviewer(api_keys=api_keys, cache_path=output_dir / "llm_review_cache.json")
    return reviewer.review_conversations(grouped_records, feature_records)


def discover_groq_keys(env_values: dict[str, str]) -> list[str]:
    keys: list[str] = []
    matched_names = sorted(
        (name for name in env_values if re.fullmatch(r"GROQ_API_KEY(?:_?\d+)?", name)),
        key=_groq_key_sort_key,
    )
    for name in matched_names:
        value = env_values.get(name)
        if value:
            keys.append(value)

    combined = env_values.get("GROQ_API_KEYS")
    if combined:
        keys.extend([item.strip() for item in combined.split(",") if item.strip()])

    deduped = []
    seen = set()
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def _groq_key_sort_key(name: str) -> tuple[int, int]:
    if name == "GROQ_API_KEY":
        return (0, 0)
    match = re.fullmatch(r"GROQ_API_KEY_?(\d+)", name)
    if match:
        return (1, int(match.group(1)))
    return (2, 0)


def build_prompt(grouped_record: GroupedConversationRecord, feature_record: ConversationFeatureRecord) -> str:
    transcript_lines = []
    for message in grouped_record.messages:
        text = (message.cleanText or message.rawText).strip()
        if len(text) > 1000:
            text = text[:1000] + "..."
        transcript_lines.append(f"[{message.timestamp}] [{message.sender.upper()}] [{message.messageType}] {text}")

    transcript = "\n".join(transcript_lines)
    observable_meta = {
        "event_types": feature_record.conversationMeta.eventTypes,
        "link_click_count": feature_record.conversationMeta.linkClickCount,
        "product_click_count": feature_record.conversationMeta.productClickCount,
        "feedback_click_count": feature_record.conversationMeta.feedbackClickCount,
        "product_view_count": feature_record.conversationMeta.productViewCount,
        "login_click_count": feature_record.conversationMeta.loginClickCount,
        "has_whatsapp_handoff": feature_record.conversationMeta.hasWhatsAppHandoff,
        "contains_product_links": feature_record.conversationMeta.containsProductLinks,
        "num_recommendations_observed": feature_record.conversationMeta.numRecommendationsObserved,
    }

    schema = (
        "{\n"
        '  "initialIntent": "order_support | product_discovery | product_page_question | greeting | other",\n'
        '  "languageStyle": "english | hinglish | mixed | other | unknown",\n'
        '  "hasSafetySensitiveContext": true,\n'
        '  "userRepetition": false,\n'
        '  "frustrated": false,\n'
        '  "assistantRepetition": false,\n'
        '  "assistantShortAnswer": false,\n'
        '  "assistantEvasiveAnswer": false,\n'
        '  "containsOrderInstructions": false,\n'
        '  "containsSafetyDisclaimer": false,\n'
        '  "containsPossibleClaimRisk": false,\n'
        '  "recommendationGiven": false,\n'
        '  "recommendationRelevant": false,\n'
        '  "badRecommendation": false,\n'
        '  "success": false,\n'
        '  "dropOff": false,\n'
        '  "unresolved": true,\n'
        '  "primaryProblem": "order_friction | login_loop | bad_recommendation | risk_flagged | unresolved_need | none",\n'
        '  "conversationOutcome": "resolved | unresolved | drop_off | frustrated | order_friction | login_loop | bad_recommendation | risk_flagged | neutral",\n'
        '  "userNeverResponded": false,\n'
        '  "userDidNotProvideRequiredInfo": false,\n'
        '  "userAskedUnrelatedQuestions": false,\n'
        '  "userOrderingMistake": false,\n'
        '  "userDidNotFollowInstructions": false,\n'
        '  "assistantHallucinating": false,\n'
        '  "assistantWrongProductSuggestion": false,\n'
        '  "assistantHealthClaimWithoutDisclaimer": false,\n'
        '  "assistantFailedEscalation": false,\n'
        '  "assistantNoRecommendationWhenNeeded": false,\n'
        '  "userEngaged": false,\n'
        '  "recommendationConverted": false,\n'
        '  "problemCouldBeResolved": false,\n'
        '  "qualityDimensions": {"accuracy": 0, "relevance": 0, "clarity": 0, "helpfulness": 0, "tone": 0, "efficiency": 0, "escalationHandling": 0},\n'
        '  "escalation": {"needed": false, "triggered": false, "resolved": false, "timeToEscalationSeconds": 0},\n'
        '  "resolutionBlockers": {"rootCause": "", "assistantMitigation": ""},\n'
        '  "issues": ["short_snake_case_tags"],\n'
        '  "summary": "one short factual sentence",\n'
        '  "feedbackText": "one short dashboard-friendly explanation"\n'
        "}"
    )

    return (
        "<task>\n"
        "Analyze one assistant conversation.\n"
        "Use transcript as source of truth and metadata only as support.\n"
        "Return ONLY valid JSON using the schema.\n"
        "For qualityDimensions, use integer scores in [-2, 2].\n"
        "</task>\n\n"
        "<rules>\n"
        "- Be conservative. If unclear, choose neutral labels.\n"
        "- recommendationGiven=true only for product suggestions, not login/help links.\n"
        "- If unresolved with repeated login redirection, set primaryProblem=login_loop.\n"
        "- If unresolved order help without login loop, set primaryProblem=order_friction.\n"
        "- If success=false and unresolved=false, include an explicit issue tag: compliance_violation, safety_risk, or policy_violation. If no explicit reason exists, prefer success=true.\n"
        "- If conversationOutcome=resolved and unresolved=false, default success=true unless compliance_violation/safety_risk/policy_violation is present.\n"
        "- If recommendationConverted=true, set success=true and unresolved=false unless compliance_violation/safety_risk/policy_violation is present.\n"
        "- If assistantHealthClaimWithoutDisclaimer=true, set success=false and include issue health_claim_without_disclaimer.\n"
        "- If the assistant asks the user to consult a doctor or shares doctor contact details (WhatsApp/call/email), do not mark containsPossibleClaimRisk=true for that handoff alone.\n"
        "- In those doctor-handoff cases, avoid primaryProblem=risk_flagged unless there is separate explicit unsafe medical advice or unverifiable cure/efficacy claims.\n"
        "- If transcript includes WhatsApp/doctor handoff instruction or handoff link click, set escalation.triggered=true.\n"
        "- If dropOff=true but response was relevant/helpful and user got actionable next step, do not force unresolved=true.\n"
        "- escalation.needed=true when unresolved/frustrated loops suggest human intervention.\n"
        "- Set escalationHandling low when escalation was needed but not triggered.\n"
        "</rules>\n\n"
        "<format>\n"
        f"{schema}\n"
        "</format>\n\n"
        "<metadata>\n"
        f'conversationId={grouped_record.conversationId}\n'
        f'widgetId={grouped_record.widgetId}\n'
        f'brandName={grouped_record.brandName}\n'
        f"structure={json.dumps(asdict(feature_record.structure), ensure_ascii=True)}\n"
        f"observableMeta={json.dumps(observable_meta, ensure_ascii=True)}\n"
        "</metadata>\n\n"
        "<transcript>\n"
        f"{transcript}\n"
        "</transcript>"
    )


def parse_review_json(content: str) -> ReviewResult:
    data = try_parse_json(content)
    if data is None:
        raise ValueError(f"Could not parse JSON from model output: {content[:300]}")
    return parse_review_data(data)


def parse_review_data(data: dict[str, Any]) -> ReviewResult:
    issues = get_review_value(data, "issues", "issues", [])
    if isinstance(issues, str):
        issues = [issues]
    if not isinstance(issues, list):
        issues = [str(issues)]

    primary_problem = normalize_optional_string(get_review_value(data, "primaryProblem", "primary_problem"))
    if primary_problem == "none":
        primary_problem = None

    summary = normalize_optional_string(get_review_value(data, "summary", "summary")) or "No summary returned."
    feedback_text = normalize_optional_string(get_review_value(data, "feedbackText", "feedback_text")) or summary

    quality_data = get_review_value(data, "qualityDimensions", "quality_dimensions", {})
    escalation_data = get_review_value(data, "escalation", "_escalation", {})
    blockers_data = get_review_value(data, "resolutionBlockers", "resolution_blockers", {})

    if not isinstance(quality_data, dict):
        quality_data = {}
    if not isinstance(escalation_data, dict):
        escalation_data = {}
    if not isinstance(blockers_data, dict):
        blockers_data = {}

    quality_dimensions = QualityDimensions(
        accuracy=clamp_dimension(quality_data.get("accuracy", 0)),
        relevance=clamp_dimension(quality_data.get("relevance", 0)),
        clarity=clamp_dimension(quality_data.get("clarity", 0)),
        helpfulness=clamp_dimension(quality_data.get("helpfulness", 0)),
        tone=clamp_dimension(quality_data.get("tone", 0)),
        efficiency=clamp_dimension(quality_data.get("efficiency", 0)),
        escalation_handling=clamp_dimension(quality_data.get("escalationHandling", quality_data.get("escalation_handling", 0))),
    )

    resolution_blockers: dict[str, str] = {}
    for key, value in blockers_data.items():
        text = normalize_optional_string(value)
        if text is not None:
            resolution_blockers[str(key)] = text

    return ReviewResult(
        initial_intent=normalize_enum(
            get_review_value(data, "initialIntent", "initial_intent"),
            {"order_support", "product_discovery", "product_page_question", "greeting", "other"},
            "other",
        ),
        language_style=normalize_enum(
            get_review_value(data, "languageStyle", "language_style"),
            {"english", "hinglish", "mixed", "other", "unknown"},
            "unknown",
        ),
        has_safety_sensitive_context=bool(get_review_value(data, "hasSafetySensitiveContext", "has_safety_sensitive_context", False)),
        user_repetition=bool(get_review_value(data, "userRepetition", "user_repetition", False)),
        frustrated=bool(get_review_value(data, "frustrated", "frustrated", False)),
        assistant_repetition=bool(get_review_value(data, "assistantRepetition", "assistant_repetition", False)),
        assistant_short_answer=bool(get_review_value(data, "assistantShortAnswer", "assistant_short_answer", False)),
        assistant_evasive_answer=bool(get_review_value(data, "assistantEvasiveAnswer", "assistant_evasive_answer", False)),
        contains_order_instructions=bool(get_review_value(data, "containsOrderInstructions", "contains_order_instructions", False)),
        contains_safety_disclaimer=bool(get_review_value(data, "containsSafetyDisclaimer", "contains_safety_disclaimer", False)),
        contains_possible_claim_risk=bool(get_review_value(data, "containsPossibleClaimRisk", "contains_possible_claim_risk", False)),
        recommendation_given=bool(get_review_value(data, "recommendationGiven", "recommendation_given", False)),
        recommendation_relevant=bool(get_review_value(data, "recommendationRelevant", "recommendation_relevant", False)),
        bad_recommendation=bool(get_review_value(data, "badRecommendation", "bad_recommendation", False)),
        success=bool(get_review_value(data, "success", "success", False)),
        drop_off=bool(get_review_value(data, "dropOff", "drop_off", False)),
        unresolved=bool(get_review_value(data, "unresolved", "unresolved", False)),
        primary_problem=primary_problem,
        conversation_outcome=normalize_enum(
            get_review_value(data, "conversationOutcome", "conversation_outcome"),
            {
                "resolved",
                "unresolved",
                "drop_off",
                "frustrated",
                "order_friction",
                "login_loop",
                "bad_recommendation",
                "risk_flagged",
                "neutral",
            },
            "neutral",
        ),
        issues=[str(item).strip() for item in issues if str(item).strip()],
        summary=summary,
        feedback_text=feedback_text,
        user_never_responded=bool(get_review_value(data, "userNeverResponded", "user_never_responded", False)),
        user_did_not_provide_required_info=bool(get_review_value(data, "userDidNotProvideRequiredInfo", "user_did_not_provide_required_info", False)),
        user_asked_unrelated_questions=bool(get_review_value(data, "userAskedUnrelatedQuestions", "user_asked_unrelated_questions", False)),
        user_ordering_mistake=bool(get_review_value(data, "userOrderingMistake", "user_ordering_mistake", False)),
        user_did_not_follow_instructions=bool(get_review_value(data, "userDidNotFollowInstructions", "user_did_not_follow_instructions", False)),
        assistant_hallucinating=bool(get_review_value(data, "assistantHallucinating", "assistant_hallucinating", False)),
        assistant_wrong_product_suggestion=bool(get_review_value(data, "assistantWrongProductSuggestion", "assistant_wrong_product_suggestion", False)),
        assistant_health_claim_without_disclaimer=bool(get_review_value(data, "assistantHealthClaimWithoutDisclaimer", "assistant_health_claim_without_disclaimer", False)),
        assistant_failed_escalation=bool(get_review_value(data, "assistantFailedEscalation", "assistant_failed_escalation", False)),
        assistant_no_recommendation_when_needed=bool(get_review_value(data, "assistantNoRecommendationWhenNeeded", "assistant_no_recommendation_when_needed", False)),
        user_engaged=bool(get_review_value(data, "userEngaged", "user_engaged", False)),
        recommendation_converted=bool(get_review_value(data, "recommendationConverted", "recommendation_converted", False)),
        problem_could_be_resolved=bool(get_review_value(data, "problemCouldBeResolved", "problem_could_be_resolved", False)),
        quality_dimensions=quality_dimensions,
        escalation_needed=parse_optional_bool(escalation_data.get("needed")),
        escalation_triggered=parse_optional_bool(escalation_data.get("triggered")),
        escalation_resolved=parse_optional_bool(escalation_data.get("resolved")),
        time_to_escalation_seconds=parse_optional_int(escalation_data.get("timeToEscalationSeconds")),
        resolution_blockers=resolution_blockers,
    )


def get_review_value(data: dict[str, Any], camel_key: str, snake_key: str, default: Any = None) -> Any:
    if camel_key in data:
        return data[camel_key]
    if snake_key in data:
        return data[snake_key]
    return default


def finalize_review_result(
    grouped_record: GroupedConversationRecord,
    feature_record: ConversationFeatureRecord,
    review: ReviewResult,
) -> ReviewResult:
    full_user_text = " ".join(
        message.cleanText
        for message in grouped_record.messages
        if message.sender == "user" and message.messageType == "text" and message.cleanText
    )
    full_agent_text = " ".join(
        message.cleanText
        for message in grouped_record.messages
        if message.sender == "agent" and message.messageType == "text" and message.cleanText
    )
    combined_text = f"{full_user_text} {full_agent_text}".strip()
    handoff_detected = has_doctor_or_whatsapp_handoff(grouped_record)
    actionable_next_step = has_actionable_next_step(full_agent_text)

    if review.initial_intent == "other":
        review.initial_intent = infer_initial_intent(full_user_text, feature_record)

    if review.language_style == "unknown" and full_user_text.strip():
        review.language_style = detect_language_style(full_user_text)

    if not review.recommendation_given:
        review.recommendation_given = (
            feature_record.conversationMeta.containsProductLinks
            or feature_record.conversationMeta.numRecommendationsObserved > 0
        )

    if review.recommendation_given and not review.recommendation_relevant and review.bad_recommendation:
        review.recommendation_relevant = False
    elif not review.recommendation_given:
        review.recommendation_relevant = False

    if not review.contains_order_instructions:
        review.contains_order_instructions = infer_contains_order_instructions(combined_text)

    if not review.has_safety_sensitive_context:
        review.has_safety_sensitive_context = infer_safety_sensitive_context(full_user_text)

    # Relax claim-risk classification when the assistant safely hands off to doctor support.
    if (
        review.contains_possible_claim_risk
        and handoff_detected
        and not has_explicit_unsafe_medical_claim(full_agent_text)
    ):
        review.contains_possible_claim_risk = False
        if review.primary_problem == "risk_flagged":
            review.primary_problem = None
        review.issues = [
            issue
            for issue in review.issues
            if issue not in {"health_claim_without_disclaimer", "compliance_violation", "safety_risk"}
        ]

    # Strict compliance override.
    if review.assistant_health_claim_without_disclaimer:
        review.success = False
        if "health_claim_without_disclaimer" not in review.issues:
            review.issues.append("health_claim_without_disclaimer")
        if "compliance_violation" not in review.issues:
            review.issues.append("compliance_violation")

    if review.success:
        review.unresolved = False
        if review.conversation_outcome in {"neutral", "unresolved", "drop_off", "frustrated"}:
            review.conversation_outcome = "resolved"

    if review.bad_recommendation:
        review.recommendation_given = True
        review.recommendation_relevant = False
        if review.primary_problem is None:
            review.primary_problem = "bad_recommendation"

    if review.primary_problem is None:
        review.primary_problem = infer_primary_problem(review, feature_record)

    if review.unresolved and not review.success and review.conversation_outcome == "neutral":
        if review.primary_problem == "order_friction":
            review.conversation_outcome = "order_friction"
        elif review.primary_problem == "login_loop":
            review.conversation_outcome = "login_loop"
        elif review.bad_recommendation:
            review.conversation_outcome = "bad_recommendation"
        elif review.frustrated:
            review.conversation_outcome = "frustrated"
        elif review.drop_off:
            review.conversation_outcome = "drop_off"
        else:
            review.conversation_outcome = "unresolved"

    if review.conversation_outcome == "neutral":
        review.conversation_outcome = infer_conversation_outcome(review)

    if review.language_style == "unknown" and not full_user_text.strip():
        review.language_style = "unknown"
    elif review.language_style == "unknown":
        review.language_style = "other"

    if review.initial_intent == "other" and full_user_text.strip().lower() in {"hi", "hello", "hey", "hii"}:
        review.initial_intent = "greeting"

    if not review.feedback_text:
        review.feedback_text = review.summary

    if _quality_dimensions_are_default(review.quality_dimensions):
        review.quality_dimensions = infer_quality_dimensions(review)

    # Conservative escalation defaults for production visibility.
    if review.escalation_needed is None and (review.unresolved or review.frustrated):
        review.escalation_needed = True
    if review.escalation_triggered is None:
        review.escalation_triggered = bool(feature_record.conversationMeta.hasWhatsAppHandoff)
    if handoff_detected:
        review.escalation_triggered = True
    if review.escalation_resolved is None:
        review.escalation_resolved = bool(review.success and review.escalation_triggered)
    if review.escalation_needed is False and review.escalation_triggered:
        review.escalation_needed = True

    # Drop-off decoupling: good actionable assistance with user drop-off is not always unresolved.
    if review.drop_off and review.recommendation_relevant and actionable_next_step and not is_explicit_failure_state(review):
        review.unresolved = False

    # Success/unresolved and outcome consistency rules.
    if review.recommendation_converted and not is_explicit_failure_state(review):
        review.success = True
        review.unresolved = False

    if review.conversation_outcome == "resolved" and not review.unresolved and not is_explicit_failure_state(review):
        review.success = True

    if not review.success and not review.unresolved and not has_explicit_failure_reason(review):
        review.success = True

    if review.success:
        review.unresolved = False
        if review.conversation_outcome in {"neutral", "unresolved", "drop_off", "frustrated"}:
            review.conversation_outcome = "resolved"

    if not review.resolution_blockers and review.unresolved:
        review.resolution_blockers = {
            "rootCause": "unknown",
            "assistantMitigation": "not_clear",
        }

    return review


def _quality_dimensions_are_default(dimensions: QualityDimensions) -> bool:
    return (
        dimensions.accuracy == 0
        and dimensions.relevance == 0
        and dimensions.clarity == 0
        and dimensions.helpfulness == 0
        and dimensions.tone == 0
        and dimensions.efficiency == 0
        and dimensions.escalation_handling == 0
    )


def infer_quality_dimensions(review: ReviewResult) -> QualityDimensions:
    """Infer quality dimensions conservatively when legacy cache lacks explicit values."""
    accuracy = 1
    relevance = 1
    clarity = 1
    helpfulness = 1
    tone = 1
    efficiency = 1
    escalation = 0

    if review.assistant_hallucinating:
        accuracy -= 2
    if review.bad_recommendation or review.assistant_wrong_product_suggestion:
        relevance -= 2
    if review.assistant_short_answer or review.assistant_evasive_answer:
        clarity -= 1
        helpfulness -= 1
    if review.frustrated:
        tone -= 2
    if review.assistant_repetition or review.drop_off:
        efficiency -= 2
    if review.unresolved:
        helpfulness -= 2
    if review.success:
        helpfulness += 1
        relevance += 1
        efficiency += 1

    if review.escalation_needed and not review.escalation_triggered:
        escalation = -2
    elif review.escalation_triggered and review.escalation_resolved:
        escalation = 2
    elif review.escalation_triggered:
        escalation = 0

    return QualityDimensions(
        accuracy=max(-2, min(2, accuracy)),
        relevance=max(-2, min(2, relevance)),
        clarity=max(-2, min(2, clarity)),
        helpfulness=max(-2, min(2, helpfulness)),
        tone=max(-2, min(2, tone)),
        efficiency=max(-2, min(2, efficiency)),
        escalation_handling=max(-2, min(2, escalation)),
    )


def normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_enum(value: Any, allowed: set[str], default: str) -> str:
    normalized = str(value or "").strip().lower().replace(" ", "_")
    return normalized if normalized in allowed else default


def clamp_dimension(value: Any) -> int:
    parsed = parse_optional_int(value)
    if parsed is None:
        return 0
    return max(-2, min(2, parsed))


def parse_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def infer_initial_intent(
    full_user_text: str,
    feature_record: ConversationFeatureRecord,
) -> str:
    lowered = full_user_text.strip().lower()
    if not lowered:
        return "other"
    if lowered in {"hi", "hello", "hey", "hii", "namaste"}:
        return "greeting"
    if any(keyword in lowered for keyword in ORDER_KEYWORDS):
        return "order_support"
    if has_slug_like_token(lowered) or any(lowered.startswith(prefix) for prefix in PRODUCT_PAGE_PREFIXES):
        return "product_page_question"
    if any(keyword in lowered for keyword in PRODUCT_DISCOVERY_KEYWORDS):
        return "product_discovery"
    if feature_record.conversationMeta.numRecommendationsObserved > 0 or feature_record.conversationMeta.containsProductLinks:
        return "product_discovery"
    return "other"


def infer_contains_order_instructions(combined_text: str) -> bool:
    lowered = combined_text.lower()
    order_instruction_hints = (
        "order number",
        "phone number",
        "email address",
        "registered mobile",
        "track your order",
        "track order",
        "refund",
        "return request",
        "cancel order",
        "delivery status",
        "shipment status",
    )
    return any(hint in lowered for hint in order_instruction_hints)


def infer_safety_sensitive_context(user_text: str) -> bool:
    lowered = user_text.lower()
    sensitive_hints = (
        "pregnant",
        "pregnancy",
        "breastfeeding",
        "pcos",
        "pcod",
        "thyroid",
        "diabetes",
        "diabetic",
        "blood sugar",
        "blood pressure",
        "doctor",
        "safe for",
        "can i take",
        "can i use",
        "side effect",
        "dosage",
    )
    return any(hint in lowered for hint in sensitive_hints)


def infer_primary_problem(review: ReviewResult, feature_record: ConversationFeatureRecord) -> str | None:
    if feature_record.conversationMeta.loginClickCount >= 2:
        return "login_loop"
    if review.bad_recommendation:
        return "bad_recommendation"
    if review.contains_possible_claim_risk and has_strong_claim_risk_evidence(review):
        return "risk_flagged"
    if review.unresolved and review.contains_order_instructions:
        return "order_friction"
    if review.unresolved:
        return "unresolved_need"
    return None


def infer_conversation_outcome(review: ReviewResult) -> str:
    if review.success:
        return "resolved"
    if review.primary_problem == "login_loop":
        return "login_loop"
    if review.primary_problem == "order_friction":
        return "order_friction"
    if review.primary_problem == "bad_recommendation":
        return "bad_recommendation"
    if review.primary_problem == "risk_flagged":
        return "risk_flagged"
    if review.frustrated:
        return "frustrated"
    if review.drop_off:
        return "drop_off"
    if review.unresolved:
        return "unresolved"
    return "neutral"


def has_doctor_or_whatsapp_handoff(grouped_record: GroupedConversationRecord) -> bool:
    for message in grouped_record.messages:
        text = (message.cleanText or message.rawText or "").lower()
        if "whatsapp" in text or "wa.me" in text or "api.whatsapp.com" in text:
            return True
        if message.sender == "agent" and (
            "consult" in text and "doctor" in text
            or "contact our doctor" in text
            or "contact doctor" in text
        ):
            return True
    return False


def has_explicit_unsafe_medical_claim(agent_text: str) -> bool:
    lowered = agent_text.lower()
    unsafe_claim_hints = (
        "guaranteed cure",
        "cure",
        "treats",
        "reverses",
        "will fix",
        "clinically proven",
        "100%",
    )
    return any(hint in lowered for hint in unsafe_claim_hints)


def has_actionable_next_step(agent_text: str) -> bool:
    lowered = agent_text.lower()
    actionable_hints = (
        "click",
        "link",
        "whatsapp",
        "contact",
        "call",
        "order number",
        "track order",
        "checkout",
        "add to cart",
    )
    return any(hint in lowered for hint in actionable_hints)


def has_explicit_failure_reason(review: ReviewResult) -> bool:
    explicit_tags = {"compliance_violation", "safety_risk", "policy_violation"}
    return any(issue in explicit_tags for issue in review.issues)


def is_explicit_failure_state(review: ReviewResult) -> bool:
    if has_explicit_failure_reason(review):
        return True
    if review.assistant_health_claim_without_disclaimer:
        return True
    return False


def has_strong_claim_risk_evidence(review: ReviewResult) -> bool:
    if review.assistant_health_claim_without_disclaimer:
        return True
    if "health_claim_without_disclaimer" in review.issues:
        return True
    return review.has_safety_sensitive_context


def try_parse_json(content: str) -> dict[str, Any] | None:
    raw = content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "rate limit" in message or "429" in message or "too many requests" in message


def parse_retry_after_seconds(message: str) -> float | None:
    match = re.search(r"try again in (\d+)m(\d+(?:\.\d+)?)s", message, flags=re.IGNORECASE)
    if match:
        minutes = float(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds

    match = re.search(r"try again in (\d+(?:\.\d+)?)s", message, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def apply_review_result(feature_record: ConversationFeatureRecord, review: ReviewResult) -> None:
    llm = feature_record.llmReview
    llm.completed = True
    llm.provider = "groq"
    llm.model = MODEL
    llm.initialIntent = review.initial_intent
    llm.languageStyle = review.language_style
    llm.hasSafetySensitiveContext = review.has_safety_sensitive_context
    llm.userRepetition = review.user_repetition
    llm.frustrated = review.frustrated
    llm.assistantRepetition = review.assistant_repetition
    llm.assistantShortAnswer = review.assistant_short_answer
    llm.assistantEvasiveAnswer = review.assistant_evasive_answer
    llm.containsOrderInstructions = review.contains_order_instructions
    llm.containsSafetyDisclaimer = review.contains_safety_disclaimer
    llm.containsPossibleClaimRisk = review.contains_possible_claim_risk
    llm.recommendationGiven = review.recommendation_given
    llm.recommendationRelevant = review.recommendation_relevant
    llm.badRecommendation = review.bad_recommendation
    llm.success = review.success
    llm.dropOff = review.drop_off
    llm.unresolved = review.unresolved
    llm.primaryProblem = review.primary_problem
    llm.conversationOutcome = review.conversation_outcome
    llm.issues = review.issues
    llm.summary = review.summary
    llm.feedbackText = review.feedback_text
    llm.userNeverResponded = review.user_never_responded
    llm.userDidNotProvideRequiredInfo = review.user_did_not_provide_required_info
    llm.userAskedUnrelatedQuestions = review.user_asked_unrelated_questions
    llm.userOrderingMistake = review.user_ordering_mistake
    llm.userDidNotFollowInstructions = review.user_did_not_follow_instructions
    llm.assistantHallucinating = review.assistant_hallucinating
    llm.assistantWrongProductSuggestion = review.assistant_wrong_product_suggestion
    llm.assistantHealthClaimWithoutDisclaimer = review.assistant_health_claim_without_disclaimer
    llm.assistantFailedEscalation = review.assistant_failed_escalation
    llm.assistantNoRecommendationWhenNeeded = review.assistant_no_recommendation_when_needed
    llm.userEngaged = review.user_engaged
    llm.recommendationConverted = review.recommendation_converted
    llm.problemCouldBeResolved = review.problem_could_be_resolved

    llm.qualityDimensions = review.quality_dimensions
    llm.escalationNeeded = bool(review.escalation_needed)
    llm.escalationTriggered = bool(review.escalation_triggered)
    llm.escalationResolved = bool(review.escalation_resolved)
    llm.timeToEscalationSeconds = review.time_to_escalation_seconds
    llm.resolutionBlockers = review.resolution_blockers

    recalculate_score(feature_record)


def recalculate_score(feature_record: ConversationFeatureRecord) -> None:
    score = 0
    llm = feature_record.llmReview
    meta = feature_record.conversationMeta

    if meta.linkClickCount > 0 and llm.primaryProblem != "login_loop":
        score += 2
    if meta.productClickCount > 0:
        score += 2
    if llm.recommendationGiven and llm.recommendationRelevant and meta.productViewCount > 0:
        score += 2

    if llm.userEngaged:
        score += 1
    if llm.recommendationConverted:
        score += 2

    if llm.dropOff:
        score -= 1
    if meta.feedbackClickCount > 0:
        score -= 2
    if llm.assistantRepetition:
        score -= 1
    if llm.badRecommendation:
        score -= 1
    if llm.frustrated:
        score -= 2
    if llm.unresolved:
        score -= 2
    if llm.primaryProblem in {"login_loop", "order_friction", "unresolved_need"}:
        score -= 1

    if llm.assistantHallucinating:
        score -= 3
    if llm.assistantWrongProductSuggestion:
        score -= 2
    if llm.assistantHealthClaimWithoutDisclaimer:
        score -= 3
    if llm.assistantFailedEscalation:
        score -= 2
    if llm.assistantNoRecommendationWhenNeeded:
        score -= 1
    if llm.problemCouldBeResolved and llm.unresolved:
        score -= 1

    # Blend quality dimensions into score in a bounded way.
    dims = llm.qualityDimensions
    quality_delta = (
        dims.accuracy
        + dims.relevance
        + dims.clarity
        + dims.helpfulness
        + dims.tone
        + dims.efficiency
        + dims.escalation_handling
    )
    score += max(-3, min(3, round(quality_delta / 4)))

    user_penalty = 0
    if llm.userNeverResponded:
        user_penalty += 1
    if llm.userDidNotProvideRequiredInfo:
        user_penalty += 1
    if llm.userAskedUnrelatedQuestions:
        user_penalty += 1
    if llm.userOrderingMistake:
        user_penalty += 1
    if llm.userDidNotFollowInstructions:
        user_penalty += 1
    score -= min(user_penalty, 2)

    feature_record.derivedMetrics.score = score
    if score >= 2:
        feature_record.derivedMetrics.quality = "good"
    elif score >= 0:
        feature_record.derivedMetrics.quality = "neutral"
    else:
        feature_record.derivedMetrics.quality = "bad"
