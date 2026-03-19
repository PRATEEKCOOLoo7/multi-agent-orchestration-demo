# Multi-Agent Orchestration Demo

A production-pattern multi-agent system built with **OpenClaw** demonstrating autonomous agent coordination, task delegation, and handoffs for financial research and advisory workflows.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Research    │────▶│  Analysis    │────▶│  Outreach   │────▶│   Content    │
│  Agent       │     │  Agent       │     │  Agent      │     │   Agent      │
│              │     │              │     │             │     │              │
│ • Market data│     │ • Scoring    │     │ • Lead      │     │ • Marketing  │
│ • News feed  │     │ • Risk eval  │     │   nurturing │     │   copy       │
│ • SEC filings│     │ • Portfolio  │     │ • Personal- │     │ • Reports    │
│              │     │   recommend. │     │   ization   │     │ • Summaries  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
       │                    │                    │                    │
       └────────────────────┴────────────────────┴────────────────────┘
                          Shared State & Message Bus
                     (Coordination / Delegation / Handoffs)
```

## Key Features

- **Multi-Agent Coordination**: Agents communicate via a shared message bus, passing structured payloads between stages
- **Autonomous Delegation**: Each agent decides when its task is complete and delegates to the next agent in the pipeline
- **Proactive Scheduling**: Agents can be triggered on cron schedules without manual prompting
- **Task Handoffs**: Structured handoff protocol ensures context is preserved across agent boundaries
- **Quality Gate**: Every agent output passes through a validation layer before handoff
- **Fallback & Escalation**: Low-confidence outputs trigger human-in-the-loop escalation

## Tech Stack

- **OpenClaw** — Agent orchestration framework
- **LangChain** — LLM integration and chain composition
- **OpenAI GPT-4 / Claude** — Language model backends
- **Pinecone** — Vector storage for RAG context
- **Redis** — Message bus and state management
- **APScheduler** — Proactive agent scheduling

## Project Structure

```
multi-agent-orchestration-demo/
├── README.md
├── requirements.txt
├── config.py
├── main.py                    # Entry point & scheduler
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Abstract base with delegation protocol
│   ├── research_agent.py      # Market data & news gathering
│   ├── analysis_agent.py      # Scoring & portfolio recommendations
│   ├── outreach_agent.py      # Personalized lead nurturing
│   └── content_agent.py       # Marketing copy & report generation
├── orchestrator/
│   ├── __init__.py
│   ├── coordinator.py         # Multi-agent coordination engine
│   ├── message_bus.py         # Inter-agent communication
│   └── scheduler.py           # Proactive cron-based triggers
├── quality_gate/
│   ├── __init__.py
│   ├── validator.py           # Output validation before handoff
│   ├── hallucination_check.py # Factual grounding verification
│   └── escalation.py          # Low-confidence fallback handler
├── models/
│   ├── __init__.py
│   └── schemas.py             # Pydantic models for agent payloads
└── tests/
    ├── test_agents.py
    ├── test_orchestrator.py
    └── test_quality_gate.py
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/multi-agent-orchestration-demo.git
cd multi-agent-orchestration-demo
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_key
export PINECONE_API_KEY=your_key

# Run the demo
python main.py

# Run with proactive scheduling (agents fire on cron)
python main.py --scheduled

# Run tests
pytest tests/ -v
```

## How It Works

### 1. Research Agent
Gathers market data, news, and SEC filings for target companies. Produces a structured research payload with confidence scores.

### 2. Analysis Agent
Receives research payload, runs scoring models (risk assessment, opportunity scoring, BANT qualification), and generates portfolio recommendations with supporting evidence.

### 3. Outreach Agent
Takes analysis output and crafts personalized outreach messages tailored to each lead's profile, industry, and engagement history. Every message passes through the quality gate before delivery.

### 4. Content Agent
Generates marketing content, summary reports, and client briefs based on the full pipeline context. Handles multi-format output (email, social, PDF reports).

### Delegation Protocol

Each agent follows a structured handoff:

```python
class HandoffPayload(BaseModel):
    source_agent: str
    target_agent: str
    task_context: dict
    confidence_score: float
    requires_human_review: bool
    timestamp: datetime
```

If `confidence_score < 0.7`, the payload is routed to human review instead of the next agent.

## Design Decisions

- **Why OpenClaw over CrewAI/AutoGen**: OpenClaw's native support for structured handoff protocols and scheduling hooks made it the best fit for a production revenue workflow where reliability matters more than flexibility.
- **Message bus over direct calls**: Decoupling agents via Redis allows independent scaling and makes it easy to add new agents without changing existing ones.
- **Quality gate at every handoff**: Instead of a single gate at the end, each transition validates output — catching errors early is cheaper than catching them late.

## License

MIT
