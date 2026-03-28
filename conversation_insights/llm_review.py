from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from conversation_insights.config import PACKAGE_DIR
from conversation_insights.env_utils import load_env_file
from conversation_insights.models import ConversationFeatureRecord, GroupedConversationRecord
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
except ImportError:  # pragma: no cover - dependency may be absent until installed
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
                                "You review one ecommerce assistant conversation with high precision. "
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
            except Exception as exc:  # pragma: no cover - runtime API path
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


def maybe_run_llm_reviews(
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

    reviewer = GroqReviewer(
        api_keys=api_keys,
        cache_path=output_dir / "llm_review_cache.json",
    )
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
        transcript_lines.append(
            f"[{message.timestamp}] [{message.sender.upper()}] [{message.messageType}] {text}"
        )

    transcript = "\n".join(transcript_lines)
    observable_meta = {
        "first_user_text": feature_record.structure.first_user_text,
        "who_ended_conversation": feature_record.structure.who_ended_conversation,
        "event_types": feature_record.conversationMeta.eventTypes,
        "link_click_count": feature_record.conversationMeta.linkClickCount,
        "product_click_count": feature_record.conversationMeta.productClickCount,
        "feedback_click_count": feature_record.conversationMeta.feedbackClickCount,
        "product_view_count": feature_record.conversationMeta.productViewCount,
        "login_click_count": feature_record.conversationMeta.loginClickCount,
        "has_whatsapp_handoff": feature_record.conversationMeta.hasWhatsAppHandoff,
        "contains_product_links": feature_record.conversationMeta.containsProductLinks,
        "num_recommendations_observed": feature_record.conversationMeta.numRecommendationsObserved,
        "recommended_product_names": feature_record.conversationMeta.recommendedProductNames,
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
        '  "issues": ["short_snake_case_tags"],\n'
        '  "summary": "one short factual sentence",\n'
        '  "feedbackText": "one short dashboard-friendly explanation"\n'
        "}"
    )

    return (
        "<task>\n"
        "Analyze one assistant conversation.\n"
        "Use the transcript as the source of truth.\n"
        "Use metadata only as supporting signals.\n"
        "Be precise and conservative. If something is unclear, choose the generic label instead of inventing facts.\n"
        "Base intent on the first substantive user need, not on greeting-only openers.\n"
        "Think through the decision silently, then output JSON only.\n"
        "</task>\n\n"
        "<decision_order>\n"
        "1. Determine the user's main need from the first substantive request.\n"
        "2. Determine whether the conversation is resolved, unresolved, dropped off, frustrated, or has a clearer specific failure mode.\n"
        "3. Identify the main problem if there is one.\n"
        "4. Then fill intent, language, recommendation, safety, and behavior fields.\n"
        "</decision_order>\n\n"
        "<critical_rules>\n"
        "- Use the transcript as the primary evidence. Metadata may support, but must not override, the transcript.\n"
        "- Treat hello/hi/hey/hii/namaste as greeting only if no substantive need appears later.\n"
        "- Mark userRepetition=true when the user repeats the same need, even with slightly different wording.\n"
        "- Mark assistantRepetition=true when the assistant repeats the same instruction or substantially the same answer.\n"
        "- Mark containsOrderInstructions=true for requests to sign in, share order number, share phone/email, track order, refund, return, cancel, or other order-resolution steps.\n"
        "- recommendationGiven=true only for actual product suggestions or product-page recommendations.\n"
        "- recommendationGiven=false for login links, tracking links, help links, support links, WhatsApp handoff links, account actions, or generic navigation links.\n"
        "- recommendationRelevant=true only if an actual product recommendation clearly matches the user's need.\n"
        "- If the assistant repeatedly redirects the user to login or sign-in and the issue remains unresolved, prefer primaryProblem=login_loop.\n"
        "- If the issue is an order-flow problem without repeated login redirection, prefer primaryProblem=order_friction.\n"
        "- If primaryProblem=login_loop and the issue is unresolved, prefer conversationOutcome=login_loop unless the conversation is clearly resolved.\n"
        "- summary should be short and factual.\n"
        "- feedbackText should explain the outcome or failure mode for a dashboard reader, and should usually add slightly more context than summary.\n"
        "</critical_rules>\n\n"
        "<label_guide>\n"
        "- order_support: tracking, refund, cancel, return, delivery, order status, payment, edit order.\n"
        "- product_discovery: user wants help choosing a product for a goal, symptom, concern, or preference.\n"
        "- product_page_question: user asks about a specific named product, ingredient, usage, benefit, suitability, or shipping for a product.\n"
        "- greeting: greeting only, with no real need yet. If a real need appears later, do not use greeting.\n"
        "- success: the user's need is clearly satisfied, not just answered.\n"
        "- unresolved: the user's need still appears open by the end.\n"
        "- drop_off: the conversation ends without clear resolution or meaningful continued engagement.\n"
        "- recommendationRelevant: true only if the recommendation clearly fits the user's need.\n"
        "- containsPossibleClaimRisk: true if the assistant appears to make unsupported medical/health claims or risky suitability claims.\n"
        "- primaryProblem should be none when there is no clear dominant issue.\n"
        "- unresolved_need: use when the need remains open but there is no clearer failure mode like login_loop, order_friction, bad_recommendation, or risk_flagged.\n"
        "</label_guide>\n\n"
        "<examples>\n"
        "<example>\n"
        "User: Where is my order?\n"
        "Good labels: initialIntent=order_support.\n"
        "</example>\n"
        "<example>\n"
        "User: Which tea is best for weight loss?\n"
        "Good labels: initialIntent=product_discovery.\n"
        "</example>\n"
        "<example>\n"
        "User: What are the key ingredients in Shatavari?\n"
        "Good labels: initialIntent=product_page_question.\n"
        "</example>\n"
        "<example>\n"
        "User: Buy ka option nhi aa rha\n"
        "Good labels: languageStyle=hinglish.\n"
        "</example>\n"
        "<example>\n"
        "User: Hi. I want to lose belly fat after pregnancy.\n"
        "Good labels: initialIntent=product_discovery, not greeting.\n"
        "</example>\n"
        "<example>\n"
        "Assistant repeats: sign in to your account. User clicks login many times and says already signed in.\n"
        "Good labels: assistantRepetition=true, containsOrderInstructions=true, primaryProblem=login_loop, conversationOutcome=login_loop.\n"
        "</example>\n"
        "<example>\n"
        "Assistant shares only an account/login link or WhatsApp/support link.\n"
        "Good labels: recommendationGiven=false.\n"
        "</example>\n"
        "<example>\n"
        "User asks the same question twice with slightly different wording.\n"
        "Good labels: userRepetition=true.\n"
        "</example>\n"
        "</examples>\n\n"
        "<format>\n"
        "Return ONLY valid JSON using this exact schema:\n"
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
    first_user_text = feature_record.structure.first_user_text or ""
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

    if review.initial_intent == "other":
        review.initial_intent = infer_initial_intent(first_user_text, full_user_text, feature_record)

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

    if review.initial_intent == "other" and first_user_text.strip().lower() in {"hi", "hello", "hey", "hii"}:
        review.initial_intent = "greeting"

    if not review.feedback_text:
        review.feedback_text = review.summary

    return review


def normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_enum(value: Any, allowed: set[str], default: str) -> str:
    normalized = str(value or "").strip().lower().replace(" ", "_")
    return normalized if normalized in allowed else default


def infer_initial_intent(
    first_user_text: str,
    full_user_text: str,
    feature_record: ConversationFeatureRecord,
) -> str:
    lowered = (first_user_text or full_user_text).strip().lower()
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
    if review.contains_possible_claim_risk:
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
    feature_record.llmReview.completed = True
    feature_record.llmReview.provider = "groq"
    feature_record.llmReview.model = MODEL
    feature_record.llmReview.initialIntent = review.initial_intent
    feature_record.llmReview.languageStyle = review.language_style
    feature_record.llmReview.hasSafetySensitiveContext = review.has_safety_sensitive_context
    feature_record.llmReview.userRepetition = review.user_repetition
    feature_record.llmReview.frustrated = review.frustrated
    feature_record.llmReview.assistantRepetition = review.assistant_repetition
    feature_record.llmReview.assistantShortAnswer = review.assistant_short_answer
    feature_record.llmReview.assistantEvasiveAnswer = review.assistant_evasive_answer
    feature_record.llmReview.containsOrderInstructions = review.contains_order_instructions
    feature_record.llmReview.containsSafetyDisclaimer = review.contains_safety_disclaimer
    feature_record.llmReview.containsPossibleClaimRisk = review.contains_possible_claim_risk
    feature_record.llmReview.recommendationGiven = review.recommendation_given
    feature_record.llmReview.recommendationRelevant = review.recommendation_relevant
    feature_record.llmReview.badRecommendation = review.bad_recommendation
    feature_record.llmReview.success = review.success
    feature_record.llmReview.dropOff = review.drop_off
    feature_record.llmReview.unresolved = review.unresolved
    feature_record.llmReview.primaryProblem = review.primary_problem
    feature_record.llmReview.conversationOutcome = review.conversation_outcome
    feature_record.llmReview.issues = review.issues
    feature_record.llmReview.summary = review.summary
    feature_record.llmReview.feedbackText = review.feedback_text

    recalculate_score(feature_record)


def recalculate_score(feature_record: ConversationFeatureRecord) -> None:
    score = 0
    if feature_record.conversationMeta.linkClickCount > 0 and feature_record.llmReview.primaryProblem != "login_loop":
        score += 2
    if feature_record.conversationMeta.productClickCount > 0:
        score += 2
    if (
        feature_record.llmReview.recommendationGiven
        and feature_record.llmReview.recommendationRelevant
        and feature_record.conversationMeta.productViewCount > 0
    ):
        score += 2
    if feature_record.llmReview.dropOff:
        score -= 1
    if feature_record.conversationMeta.feedbackClickCount > 0:
        score -= 2
    if feature_record.llmReview.assistantRepetition:
        score -= 1
    if feature_record.llmReview.badRecommendation:
        score -= 1
    if feature_record.llmReview.frustrated:
        score -= 2
    if feature_record.llmReview.unresolved:
        score -= 2
    if feature_record.llmReview.primaryProblem in {"login_loop", "order_friction", "unresolved_need"}:
        score -= 1

    feature_record.derivedMetrics.score = score
    if score >= 2:
        feature_record.derivedMetrics.quality = "good"
    elif score >= 0:
        feature_record.derivedMetrics.quality = "neutral"
    else:
        feature_record.derivedMetrics.quality = "bad"
