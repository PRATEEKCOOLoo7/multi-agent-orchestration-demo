import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from agents import BaseAgent, AgentRole
from agents.research_agent import ResearchAgent
from agents.analysis_agent import AnalysisAgent
from agents.outreach_agent import OutreachAgent
from models import HandoffPayload

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """Multi-agent coordination engine.
    
    Manages the full agent pipeline: Research → Analysis → Outreach → Content.
    Handles delegation decisions, handoff routing, and human escalation.
    """

    def __init__(self):
        self.agents: dict[AgentRole, BaseAgent] = {
            AgentRole.RESEARCH: ResearchAgent(),
            AgentRole.ANALYSIS: AnalysisAgent(),
            AgentRole.OUTREACH: OutreachAgent(),
        }
        self.execution_log: list[dict[str, Any]] = []

    async def run_pipeline(
        self,
        initial_context: dict[str, Any],
        start_from: AgentRole = AgentRole.RESEARCH,
    ) -> dict[str, Any]:
        """Execute the full agent pipeline with coordinated handoffs.
        
        Args:
            initial_context: Starting context (e.g., {"company_name": "Acme Corp"})
            start_from: Which agent to begin with (default: RESEARCH)
            
        Returns:
            Final pipeline result with all agent outputs and handoff history
        """
        logger.info(f"Pipeline started at {datetime.utcnow().isoformat()}")

        current_role = start_from
        current_context = initial_context
        pipeline_results = []
        handoff_chain = []

        while current_role and current_role in self.agents:
            agent = self.agents[current_role]
            logger.info(f"Executing: {current_role.value}")

            # Run the agent (execute → validate → prepare handoff)
            handoff = await agent.run(current_context)
            handoff_chain.append(handoff.model_dump())

            # Log execution
            self._log_execution(current_role, handoff)
            pipeline_results.append({
                "agent": current_role.value,
                "output": handoff.task_context.get("output", {}),
                "confidence": handoff.confidence_score,
                "passed_quality_gate": not handoff.requires_human_review,
            })

            # Delegation decision
            if agent.should_delegate(handoff):
                logger.info(
                    f"Delegating: {current_role.value} → "
                    f"{handoff.target_agent.value}"
                )
                current_role = handoff.target_agent
                current_context = handoff.task_context
            else:
                if handoff.requires_human_review:
                    logger.warning(
                        f"Pipeline paused at {current_role.value} — "
                        f"human review required"
                    )
                else:
                    logger.info(
                        f"Pipeline complete at {current_role.value}"
                    )
                break

        return {
            "status": "complete" if not handoff.requires_human_review else "needs_review",
            "results": pipeline_results,
            "handoff_chain": handoff_chain,
            "total_agents_executed": len(pipeline_results),
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def run_single_agent(
        self,
        role: AgentRole,
        context: dict[str, Any],
    ) -> HandoffPayload:
        """Run a single agent independently (useful for retries)."""
        if role not in self.agents:
            raise ValueError(f"No agent registered for role: {role}")
        return await self.agents[role].run(context)

    def _log_execution(self, role: AgentRole, handoff: HandoffPayload):
        """Log agent execution for observability."""
        self.execution_log.append({
            "agent": role.value,
            "confidence": handoff.confidence_score,
            "requires_review": handoff.requires_human_review,
            "timestamp": handoff.timestamp.isoformat(),
            "issues": handoff.metadata.get("issues", []),
        })

    def get_execution_summary(self) -> dict[str, Any]:
        """Return a summary of all executions in the current session."""
        if not self.execution_log:
            return {"message": "No executions recorded"}

        avg_confidence = sum(
            e["confidence"] for e in self.execution_log
        ) / len(self.execution_log)

        return {
            "total_executions": len(self.execution_log),
            "average_confidence": round(avg_confidence, 3),
            "human_reviews_triggered": sum(
                1 for e in self.execution_log if e["requires_review"]
            ),
            "log": self.execution_log,
        }
