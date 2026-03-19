import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from agents import AgentRole, BaseAgent
from agents.research_agent import ResearchAgent
from agents.analysis_agent import AnalysisAgent
from agents.outreach_agent import OutreachAgent
from orchestrator import AgentCoordinator
from models import HandoffPayload, QualityGateResult


class TestQualityGate:
    """Tests for agent quality gate validation."""

    @pytest.mark.asyncio
    async def test_research_agent_validates_complete_output(self):
        agent = ResearchAgent()
        output = {
            "company_name": "Test Corp",
            "market_data": {"revenue": "10M"},
            "news_summaries": ["Company launched new product"],
            "signals": ["bullish momentum"],
            "confidence": 0.8,
        }
        result = await agent.validate_output(output)
        assert result.passed is True
        assert result.overall_score == 1.0

    @pytest.mark.asyncio
    async def test_research_agent_fails_empty_output(self):
        agent = ResearchAgent()
        output = {
            "market_data": {},
            "news_summaries": [],
            "signals": [],
            "confidence": 0.2,
        }
        result = await agent.validate_output(output)
        assert result.passed is False
        assert len(result.issues) > 0

    @pytest.mark.asyncio
    async def test_analysis_validates_score_ranges(self):
        agent = AnalysisAgent()
        output = {
            "risk_score": 15,  # Invalid: > 10
            "opportunity_score": 7,
            "bant_qualification": {
                "budget": True, "authority": True,
                "need": True, "timeline": False,
            },
            "portfolio_recommendation": "buy",
            "supporting_evidence": ["Strong growth", "Market leader"],
            "confidence": 0.8,
        }
        result = await agent.validate_output(output)
        assert result.checks["risk_score_valid"] is False

    @pytest.mark.asyncio
    async def test_analysis_flags_inconsistent_scores(self):
        agent = AnalysisAgent()
        output = {
            "risk_score": 9,
            "opportunity_score": 9,
            "bant_qualification": {
                "budget": True, "authority": True,
                "need": True, "timeline": True,
            },
            "portfolio_recommendation": "buy",
            "supporting_evidence": ["One reason"],  # Too few for high-high
            "confidence": 0.8,
        }
        result = await agent.validate_output(output)
        assert result.checks["scores_consistent"] is False
        assert result.escalate_to_human is True


class TestOutreachCompliance:
    """Tests for outreach quality gate — tone, personalization, compliance."""

    @pytest.mark.asyncio
    async def test_blocks_generic_messages(self):
        agent = OutreachAgent()
        output = {
            "lead_name": "John",
            "subject_line": "Quick question",
            "personalized_message": "I hope this finds you well. I wanted to reach out about our services.",
            "channel": "email",
            "tone_score": 0.7,
            "confidence": 0.6,
        }
        result = await agent.validate_output(output)
        assert result.checks["not_generic"] is False

    @pytest.mark.asyncio
    async def test_blocks_prohibited_financial_claims(self):
        agent = OutreachAgent()
        output = {
            "lead_name": "Jane",
            "subject_line": "Investment opportunity",
            "personalized_message": "This is a guaranteed return opportunity with risk-free investment.",
            "channel": "email",
            "tone_score": 0.7,
            "confidence": 0.6,
        }
        result = await agent.validate_output(output)
        assert result.checks["no_prohibited_claims"] is False
        assert result.escalate_to_human is True

    @pytest.mark.asyncio
    async def test_passes_personalized_compliant_message(self):
        agent = OutreachAgent()
        output = {
            "lead_name": "Sarah Chen",
            "subject_line": "AI initiative at Acme",
            "personalized_message": (
                "Sarah, noticed Acme's recent AI announcement — your portfolio "
                "analytics work aligns with what we're building at Pearl. "
                "Would you have 15 minutes to chat this week?"
            ),
            "channel": "email",
            "tone_score": 0.85,
            "confidence": 0.9,
        }
        result = await agent.validate_output(output)
        assert result.checks["not_generic"] is True
        assert result.checks["no_prohibited_claims"] is True
        assert result.checks["has_cta"] is True


class TestDelegationProtocol:
    """Tests for agent delegation and handoff logic."""

    def test_delegates_when_confident(self):
        agent = ResearchAgent()
        handoff = HandoffPayload(
            source_agent=AgentRole.RESEARCH,
            target_agent=AgentRole.ANALYSIS,
            task_context={"output": {"data": "test"}},
            confidence_score=0.85,
            requires_human_review=False,
        )
        assert agent.should_delegate(handoff) is True

    def test_blocks_delegation_low_confidence(self):
        agent = ResearchAgent()
        handoff = HandoffPayload(
            source_agent=AgentRole.RESEARCH,
            target_agent=AgentRole.ANALYSIS,
            task_context={"output": {"data": "test"}},
            confidence_score=0.3,
            requires_human_review=False,
        )
        assert agent.should_delegate(handoff) is False

    def test_blocks_delegation_when_review_needed(self):
        agent = ResearchAgent()
        handoff = HandoffPayload(
            source_agent=AgentRole.RESEARCH,
            target_agent=AgentRole.ANALYSIS,
            task_context={"output": {"data": "test"}},
            confidence_score=0.9,
            requires_human_review=True,
        )
        assert agent.should_delegate(handoff) is False
