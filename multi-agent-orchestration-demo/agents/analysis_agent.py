import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from agents import BaseAgent, AgentRole
from config import Config
from models import QualityGateResult

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial analysis agent. Given research data about a company,
perform scoring and generate portfolio recommendations.

Evaluate:
1. Risk Score (0-10): based on market volatility, news sentiment, financial health
2. Opportunity Score (0-10): based on growth signals, market position, momentum
3. BANT Qualification: Budget (bool), Authority (bool), Need (bool), Timeline (bool)
4. Portfolio Recommendation: clear action (buy/hold/sell/watch) with reasoning

Output structured JSON with: risk_score, opportunity_score, bant_qualification, 
portfolio_recommendation, supporting_evidence (list of reasons), confidence (0.0-1.0)"""),
    ("human", "Analyze this research data:\n{research_data}")
])


class AnalysisAgent(BaseAgent):
    """Runs scoring models and generates portfolio recommendations."""

    def __init__(self):
        super().__init__(
            role=AgentRole.ANALYSIS,
            next_agent=AgentRole.OUTREACH
        )
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=0.0,
        )
        self.chain = ANALYSIS_PROMPT | self.llm

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Score and analyze the research data."""
        research_data = context.get("output", context)
        logger.info(
            f"[Analysis] Scoring: {research_data.get('company_name', 'Unknown')}"
        )

        import json
        response = await self.chain.ainvoke({
            "research_data": json.dumps(research_data, default=str),
        })

        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            result = {
                "risk_score": 5.0,
                "opportunity_score": 5.0,
                "bant_qualification": {
                    "budget": False, "authority": False,
                    "need": False, "timeline": False,
                },
                "portfolio_recommendation": "watch",
                "supporting_evidence": [response.content[:200]],
                "confidence": 0.4,
            }

        result["company_name"] = research_data.get("company_name", "Unknown")
        return result

    async def validate_output(self, output: dict[str, Any]) -> QualityGateResult:
        """Validate analysis output — check scoring consistency and evidence."""
        checks = {
            "risk_score_valid": 0 <= output.get("risk_score", -1) <= 10,
            "opportunity_score_valid": 0 <= output.get("opportunity_score", -1) <= 10,
            "has_recommendation": bool(output.get("portfolio_recommendation")),
            "has_evidence": len(output.get("supporting_evidence", [])) >= 2,
            "bant_complete": all(
                k in output.get("bant_qualification", {})
                for k in ["budget", "authority", "need", "timeline"]
            ),
            "scores_consistent": self._check_score_consistency(output),
        }

        issues = []
        if not checks["risk_score_valid"]:
            issues.append("Risk score outside valid range 0-10")
        if not checks["has_evidence"]:
            issues.append("Insufficient supporting evidence (need 2+)")
        if not checks["scores_consistent"]:
            issues.append("Risk and opportunity scores appear inconsistent")

        passed_count = sum(checks.values())
        overall_score = passed_count / len(checks)

        return QualityGateResult(
            passed=all(checks.values()),
            checks=checks,
            overall_score=overall_score,
            issues=issues,
            escalate_to_human=not checks["scores_consistent"],
        )

    @staticmethod
    def _check_score_consistency(output: dict) -> bool:
        """Flag if high risk + high opportunity without explanation."""
        risk = output.get("risk_score", 0)
        opp = output.get("opportunity_score", 0)
        evidence = output.get("supporting_evidence", [])
        if risk > 7 and opp > 7 and len(evidence) < 3:
            return False
        return True
