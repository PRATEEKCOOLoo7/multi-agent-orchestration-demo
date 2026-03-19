from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    OUTREACH = "outreach"
    CONTENT = "content"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class HandoffPayload(BaseModel):
    """Structured payload passed between agents during task delegation."""

    source_agent: AgentRole
    target_agent: AgentRole
    task_context: dict[str, Any]
    confidence_score: float = Field(ge=0.0, le=1.0)
    requires_human_review: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def confidence_level(self) -> ConfidenceLevel:
        if self.confidence_score >= 0.85:
            return ConfidenceLevel.HIGH
        elif self.confidence_score >= 0.7:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW


class ResearchOutput(BaseModel):
    """Output from the Research Agent."""

    company_name: str
    market_data: dict[str, Any]
    news_summaries: list[str]
    sec_filings: list[dict[str, str]]
    signals: list[str]
    confidence_score: float = Field(ge=0.0, le=1.0)


class AnalysisOutput(BaseModel):
    """Output from the Analysis Agent."""

    risk_score: float = Field(ge=0.0, le=10.0)
    opportunity_score: float = Field(ge=0.0, le=10.0)
    bant_qualification: dict[str, bool]
    portfolio_recommendation: str
    supporting_evidence: list[str]
    confidence_score: float = Field(ge=0.0, le=1.0)


class OutreachOutput(BaseModel):
    """Output from the Outreach Agent."""

    lead_name: str
    personalized_message: str
    subject_line: str
    channel: str  # email, linkedin, etc.
    tone_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)


class ContentOutput(BaseModel):
    """Output from the Content Agent."""

    content_type: str  # blog, report, social, email
    title: str
    body: str
    target_audience: str
    confidence_score: float = Field(ge=0.0, le=1.0)


class QualityGateResult(BaseModel):
    """Result of quality gate validation."""

    passed: bool
    checks: dict[str, bool]
    overall_score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    escalate_to_human: bool = False
