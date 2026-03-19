import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from config import Config
from models import AgentRole, HandoffPayload, QualityGateResult

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base agent with built-in delegation protocol and quality gate."""

    def __init__(self, role: AgentRole, next_agent: Optional[AgentRole] = None):
        self.role = role
        self.next_agent = next_agent
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD

    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent's core task. Must be implemented by subclasses."""
        ...

    @abstractmethod
    async def validate_output(self, output: dict[str, Any]) -> QualityGateResult:
        """Validate agent output before handoff. Must be implemented by subclasses."""
        ...

    async def run(self, context: dict[str, Any]) -> HandoffPayload:
        """Full agent lifecycle: execute → validate → prepare handoff."""
        logger.info(f"[{self.role.value}] Starting execution")

        # Execute core task
        output = await self.execute(context)

        # Run quality gate
        validation = await self.validate_output(output)
        logger.info(
            f"[{self.role.value}] Quality gate: "
            f"{'PASSED' if validation.passed else 'FAILED'} "
            f"(score={validation.overall_score:.2f})"
        )

        # Determine if human review is needed
        needs_review = (
            not validation.passed
            or validation.escalate_to_human
            or validation.overall_score < self.confidence_threshold
        )

        if needs_review:
            logger.warning(
                f"[{self.role.value}] Routing to human review. "
                f"Issues: {validation.issues}"
            )

        # Build handoff payload
        handoff = HandoffPayload(
            source_agent=self.role,
            target_agent=self.next_agent or self.role,
            task_context={
                "input": context,
                "output": output,
                "validation": validation.model_dump(),
            },
            confidence_score=validation.overall_score,
            requires_human_review=needs_review,
            metadata={
                "checks_passed": sum(validation.checks.values()),
                "checks_total": len(validation.checks),
                "issues": validation.issues,
            },
            timestamp=datetime.utcnow(),
        )

        logger.info(
            f"[{self.role.value}] Handoff prepared → {self.next_agent or 'END'} "
            f"(confidence={handoff.confidence_score:.2f})"
        )

        return handoff

    def should_delegate(self, handoff: HandoffPayload) -> bool:
        """Decide whether to delegate to the next agent or escalate."""
        if handoff.requires_human_review:
            return False
        if handoff.confidence_score < self.confidence_threshold:
            return False
        return self.next_agent is not None
