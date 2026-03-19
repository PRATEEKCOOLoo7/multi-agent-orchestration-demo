import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from agents import BaseAgent, AgentRole
from config import Config
from models import QualityGateResult

logger = logging.getLogger(__name__)

RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial research agent. Given a company name, 
gather and synthesize market data, recent news, and key signals.

Output a structured JSON with:
- market_data: key financial metrics
- news_summaries: list of 3-5 recent relevant news items (1 sentence each)
- signals: list of buying/selling signals detected
- confidence: your confidence score from 0.0 to 1.0"""),
    ("human", "Research company: {company_name}\nAdditional context: {context}")
])


class ResearchAgent(BaseAgent):
    """Gathers market data, news, and SEC filings for target companies."""

    def __init__(self):
        super().__init__(
            role=AgentRole.RESEARCH,
            next_agent=AgentRole.ANALYSIS
        )
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=0.1,
        )
        self.chain = RESEARCH_PROMPT | self.llm

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Gather market research for the target company."""
        company_name = context.get("company_name", "Unknown")
        logger.info(f"[Research] Gathering data for: {company_name}")

        response = await self.chain.ainvoke({
            "company_name": company_name,
            "context": context.get("additional_context", ""),
        })

        # Parse structured output
        import json
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            result = {
                "market_data": {},
                "news_summaries": [response.content[:200]],
                "signals": [],
                "confidence": 0.5,
            }

        result["company_name"] = company_name
        return result

    async def validate_output(self, output: dict[str, Any]) -> QualityGateResult:
        """Validate research output quality before handoff."""
        checks = {
            "has_market_data": bool(output.get("market_data")),
            "has_news": len(output.get("news_summaries", [])) > 0,
            "has_signals": len(output.get("signals", [])) > 0,
            "confidence_above_minimum": output.get("confidence", 0) > 0.3,
        }

        issues = []
        if not checks["has_market_data"]:
            issues.append("No market data retrieved")
        if not checks["has_news"]:
            issues.append("No news summaries found")
        if not checks["has_signals"]:
            issues.append("No trading signals detected")

        passed_count = sum(checks.values())
        total_count = len(checks)
        overall_score = passed_count / total_count

        return QualityGateResult(
            passed=all(checks.values()),
            checks=checks,
            overall_score=overall_score,
            issues=issues,
            escalate_to_human=overall_score < 0.5,
        )
