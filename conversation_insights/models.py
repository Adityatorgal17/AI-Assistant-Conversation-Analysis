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


@dataclass(slots=True)
class QualityDimensions:
    """Multi-dimensional quality assessment for conversations (scale: -2 to +2)."""

    accuracy: int = 0
    relevance: int = 0
    clarity: int = 0
    helpfulness: int = 0
    tone: int = 0
    efficiency: int = 0
    escalation_handling: int = 0


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
class MessageFlag:
    messageId: str
    sender: str
    flag: str
    label: str
    reason: str
    severity: str = "medium"


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
    userNeverResponded: bool | None = None
    userDidNotProvideRequiredInfo: bool | None = None
    userAskedUnrelatedQuestions: bool | None = None
    userOrderingMistake: bool | None = None
    userDidNotFollowInstructions: bool | None = None
    assistantHallucinating: bool | None = None
    assistantWrongProductSuggestion: bool | None = None
    assistantHealthClaimWithoutDisclaimer: bool | None = None
    assistantFailedEscalation: bool | None = None
    assistantNoRecommendationWhenNeeded: bool | None = None
    userEngaged: bool | None = None
    recommendationConverted: bool | None = None
    problemCouldBeResolved: bool | None = None

    # New production fields
    qualityDimensions: QualityDimensions = field(default_factory=QualityDimensions)
    escalationNeeded: bool | None = None
    escalationTriggered: bool | None = None
    escalationResolved: bool | None = None
    timeToEscalationSeconds: int | None = None
    resolutionBlockers: dict[str, str] = field(default_factory=dict)
    messageFlags: list[MessageFlag] = field(default_factory=list)


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

    # New quality dimensions for dashboard-level root-cause analysis
    qualityAccuracy: int = 0
    qualityRelevance: int = 0
    qualityClarity: int = 0
    qualityHelpfulness: int = 0
    qualityTone: int = 0
    qualityEfficiency: int = 0
    qualityEscalationHandling: int = 0

    # New escalation tracking fields
    escalationNeeded: bool = False
    escalationTriggered: bool = False
    escalationResolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InsightRecommendation:
    """Actionable insight with severity and data-backed evidence."""

    title: str
    severity: str
    relatedCount: int
    percentage: float
    description: str
    rootCauses: list[str]
    suggestedActions: list[str]
    affectedCategories: list[str]

    # Intervention tracking for validation in production.
    metricBaseline: float | None = None
    metricTarget: float | None = None
    interventionEffort: str | None = None
    interventionRisk: str | None = None
    hypothesis: str | None = None

    # Provenance and evidence metadata for hybrid/second-pass insights.
    source: str = "deterministic"
    confidence: float | None = None
    whyNew: str | None = None
    evidenceMetrics: dict[str, float | int | str] = field(default_factory=dict)


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
    recommendations: list[InsightRecommendation] = field(default_factory=list)
    assistantMistakes: dict[str, int] = field(default_factory=dict)
    userMistakes: dict[str, int] = field(default_factory=dict)

    # Aggregated quality dimensions for root-cause reporting.
    qualityDimensionsAggregate: dict[str, float] = field(default_factory=dict)

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
    recommendations: list[InsightRecommendation] = field(default_factory=list)
    crossBrandFindings: dict[str, Any] = field(default_factory=dict)
    assistantMistakesGlobal: dict[str, int] = field(default_factory=dict)
    userMistakesGlobal: dict[str, int] = field(default_factory=dict)

    # Global quality-dimension root-cause summary.
    qualityDimensionsGlobal: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
