from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class MessageRecord:
    messageId: str
    sender: str
    messageType: str
    rawText: str
    cleanText: str
    eventType: str | None
    timestamp: str


@dataclass(slots=True)
class GroupedConversationRecord:
    conversationId: str
    widgetId: str
    brandName: str | None
    createdAt: str
    updatedAt: str
    messages: list[MessageRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StructureMetrics:
    num_messages: int
    num_event_messages: int
    num_user_turns: int
    num_agent_turns: int
    first_user_text: str | None
    who_ended_conversation: str


@dataclass(slots=True)
class ConversationMeta:
    eventTypes: list[str] = field(default_factory=list)
    linkClickCount: int = 0
    productClickCount: int = 0
    feedbackClickCount: int = 0
    productViewCount: int = 0
    loginClickCount: int = 0
    hasWhatsAppHandoff: bool = False
    containsProductLinks: bool = False
    numRecommendationsObserved: int = 0
    recommendedProductNames: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LLMReview:
    completed: bool = False
    provider: str | None = None
    model: str | None = None
    initialIntent: str | None = None
    languageStyle: str | None = None
    hasSafetySensitiveContext: bool | None = None
    userRepetition: bool | None = None
    frustrated: bool | None = None
    assistantRepetition: bool | None = None
    assistantShortAnswer: bool | None = None
    assistantEvasiveAnswer: bool | None = None
    containsOrderInstructions: bool | None = None
    containsSafetyDisclaimer: bool | None = None
    containsPossibleClaimRisk: bool | None = None
    recommendationGiven: bool | None = None
    recommendationRelevant: bool | None = None
    badRecommendation: bool | None = None
    success: bool | None = None
    dropOff: bool | None = None
    unresolved: bool | None = None
    primaryProblem: str | None = None
    conversationOutcome: str | None = None
    issues: list[str] = field(default_factory=list)
    summary: str | None = None
    feedbackText: str | None = None


@dataclass(slots=True)
class DerivedMetrics:
    score: int = 0
    quality: str = "neutral"


@dataclass(slots=True)
class ConversationFeatureRecord:
    conversationId: str
    widgetId: str
    brandName: str | None
    structure: StructureMetrics
    conversationMeta: ConversationMeta
    llmReview: LLMReview
    derivedMetrics: DerivedMetrics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DashboardConversationRow:
    widgetId: str
    brandName: str | None
    conversationId: str
    initialIntent: str
    numMessages: int
    numUserTurns: int
    numAgentTurns: int
    whoEndedConversation: str
    quality: str
    score: int
    success: bool
    dropOff: bool
    frustrated: bool
    unresolved: bool
    recommendationGiven: bool
    recommendationClicked: bool
    productClick: bool
    linkClick: bool
    feedbackClick: bool
    assistantRepetition: bool
    badRecommendation: bool
    possibleClaimRisk: bool
    conversationOutcome: str
    primaryProblem: str | None
    feedbackText: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WidgetInsightSummary:
    widgetId: str
    brandName: str | None
    totalConversations: int
    qualityBreakdown: dict[str, int]
    outcomeBreakdown: dict[str, int]
    topIntents: list[dict[str, int | str]]
    topProblems: list[dict[str, int | str]]
    summaryPoints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GlobalInsightSummary:
    totalConversations: int
    qualityBreakdown: dict[str, int]
    outcomeBreakdown: dict[str, int]
    topIntents: list[dict[str, int | str]]
    topProblems: list[dict[str, int | str]]
    summaryPoints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
