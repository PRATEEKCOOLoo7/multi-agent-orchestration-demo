import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from agents import BaseAgent, AgentRole
from config import Config
from models import QualityGateResult

logger = logging.getLogger(__name__)

OUTREACH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a personalized outreach agent. Given analysis data about a 
company/lead, craft a highly personalized outreach message.

Rules:
- Reference specific data points from the analysis (never generic)
- Match tone to the lead's industry and seniority
- Keep subject line under 50 characters
- Message body under 150 words
- Include a clear, low-friction CTA

Output JSON with: lead_name, subject_line, personalized_message, channel, 
tone_score (0-1), confidence (0-1)"""),
    ("human", "Analysis data:\n{analysis_data}\nLead info:\n{lead_info}")
])


class OutreachAgent(BaseAgent):
    """Crafts personalized outreach based on analysis results."""

    def __init__(self):
        super().__init__(
            role=AgentRole.OUTREACH,
            next_agent=AgentRole.CONTENT
        )
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=0.7,
        )
        self.chain = OUTREACH_PROMPT | self.llm

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate personalized outreach message."""
        analysis_data = context.get("output", context)
        lead_info = context.get("lead_info", {
            "name": "Decision Maker",
            "title": "VP of Finance",
            "company": analysis_data.get("company_name", "Target Co"),
        })

        logger.info(f"[Outreach] Crafting message for: {lead_info.get('name')}")

        import json
        response = await self.chain.ainvoke({
            "analysis_data": json.dumps(analysis_data, default=str),
            "lead_info": json.dumps(lead_info),
        })

        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            result = {
                "lead_name": lead_info.get("name", "Unknown"),
                "subject_line": "Quick question",
                "personalized_message": response.content[:300],
                "channel": "email",
                "tone_score": 0.5,
                "confidence": 0.4,
            }

        return result

    async def validate_output(self, output: dict[str, Any]) -> QualityGateResult:
        """Quality gate: tone, personalization, compliance, length."""
        message = output.get("personalized_message", "")
        subject = output.get("subject_line", "")

        checks = {
            "subject_under_50_chars": len(subject) <= 50,
            "message_under_150_words": len(message.split()) <= 160,
            "not_generic": not self._is_generic(message),
            "has_cta": self._has_call_to_action(message),
            "tone_acceptable": output.get("tone_score", 0) >= 0.6,
            "no_prohibited_claims": self._check_compliance(message),
        }

        issues = []
        if not checks["subject_under_50_chars"]:
            issues.append(f"Subject too long: {len(subject)} chars (max 50)")
        if not checks["not_generic"]:
            issues.append("Message appears generic — lacks personalization")
        if not checks["no_prohibited_claims"]:
            issues.append("Message contains prohibited financial claims")

        passed_count = sum(checks.values())
        overall_score = passed_count / len(checks)

        return QualityGateResult(
            passed=all(checks.values()),
            checks=checks,
            overall_score=overall_score,
            issues=issues,
            escalate_to_human=not checks["no_prohibited_claims"],
        )

    @staticmethod
    def _is_generic(message: str) -> bool:
        """Detect generic outreach that lacks personalization."""
        generic_phrases = [
            "i hope this finds you well",
            "i wanted to reach out",
            "i came across your profile",
            "just following up",
        ]
        message_lower = message.lower()
        return any(phrase in message_lower for phrase in generic_phrases)

    @staticmethod
    def _has_call_to_action(message: str) -> bool:
        """Check for a clear CTA."""
        cta_indicators = [
            "schedule", "book", "chat", "call", "meet", "demo",
            "15 minutes", "quick call", "let me know", "interested",
        ]
        message_lower = message.lower()
        return any(cta in message_lower for cta in cta_indicators)

    @staticmethod
    def _check_compliance(message: str) -> bool:
        """Check for prohibited financial claims."""
        prohibited = [
            "guaranteed return",
            "risk-free",
            "you will make money",
            "100% safe",
            "can't lose",
        ]
        message_lower = message.lower()
        return not any(phrase in message_lower for phrase in prohibited)
