import argparse
import asyncio
import logging
import sys

from config import Config
from orchestrator import AgentCoordinator
from agents import AgentRole

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_demo():
    """Run a single pipeline execution with a sample company."""
    coordinator = AgentCoordinator()

    # Sample input — in production this comes from CRM or scheduled triggers
    context = {
        "company_name": "Acme Financial Corp",
        "additional_context": "Mid-cap fintech company, recently announced AI initiative",
        "lead_info": {
            "name": "Sarah Chen",
            "title": "VP of Product",
            "company": "Acme Financial Corp",
            "industry": "fintech",
        },
    }

    print("\n" + "=" * 60)
    print("  Multi-Agent Orchestration Demo")
    print("  Research → Analysis → Outreach")
    print("=" * 60 + "\n")

    # Run the full pipeline
    result = await coordinator.run_pipeline(context)

    # Display results
    print(f"\nPipeline Status: {result['status']}")
    print(f"Agents Executed: {result['total_agents_executed']}")
    print()

    for step in result["results"]:
        status = "✓" if step["passed_quality_gate"] else "⚠ REVIEW"
        print(f"  [{status}] {step['agent'].upper()}")
        print(f"       Confidence: {step['confidence']:.2f}")

        # Show key outputs
        output = step["output"]
        if step["agent"] == "research":
            signals = output.get("signals", [])
            print(f"       Signals: {', '.join(signals[:3]) if signals else 'None'}")
        elif step["agent"] == "analysis":
            print(f"       Risk: {output.get('risk_score', 'N/A')}/10")
            print(f"       Opportunity: {output.get('opportunity_score', 'N/A')}/10")
            print(f"       Recommendation: {output.get('portfolio_recommendation', 'N/A')}")
        elif step["agent"] == "outreach":
            print(f"       Subject: {output.get('subject_line', 'N/A')}")
        print()

    # Execution summary
    summary = coordinator.get_execution_summary()
    print(f"Average Confidence: {summary['average_confidence']}")
    print(f"Human Reviews Triggered: {summary['human_reviews_triggered']}")

    return result


def run_scheduled():
    """Run the pipeline on a recurring schedule (proactive agents)."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    scheduler = AsyncIOScheduler()

    async def scheduled_run():
        logger.info("Scheduled pipeline execution triggered")
        coordinator = AgentCoordinator()
        # In production, this would pull from a CRM queue
        context = {
            "company_name": "Scheduled Target Co",
            "additional_context": "Automated scheduled research run",
        }
        result = await coordinator.run_pipeline(context)
        logger.info(
            f"Scheduled run complete: {result['status']} "
            f"({result['total_agents_executed']} agents)"
        )

    scheduler.add_job(
        lambda: asyncio.create_task(scheduled_run()),
        "interval",
        minutes=Config.SCHEDULE_INTERVAL_MINUTES,
    )

    print(f"\nProactive scheduler started "
          f"(every {Config.SCHEDULE_INTERVAL_MINUTES} min)")
    print("Press Ctrl+C to stop\n")

    scheduler.start()

    loop = asyncio.new_event_loop()
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        scheduler.shutdown()
        print("\nScheduler stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Orchestration Demo")
    parser.add_argument(
        "--scheduled", action="store_true",
        help="Run agents on a recurring schedule (proactive mode)",
    )
    args = parser.parse_args()

    if args.scheduled:
        run_scheduled()
    else:
        asyncio.run(run_demo())
