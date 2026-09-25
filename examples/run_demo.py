"""Runnable end-to-end demo — no API key required.

    python examples/run_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (
    Orchestrator,
    Session,
    BillingAgent,
    TechSupportAgent,
    AccountAgent,
    FallbackAgent,
    TFIDFIntentClassifier,
)

INTENT_EXAMPLES = {
    "billing_inquiry": [
        "why is my bill so high",
        "I want to pay my balance",
        "how much do I owe?",
        "there's a charge I don't recognize",
    ],
    "tech_support": [
        "my internet is down",
        "the wifi keeps disconnecting",
        "I have something",
        "my router isn't working",
    ],
    "account_management": [
        "what plan am I on",
        "I want to upgrade my plan",
        "update my account info",
    ],
}


def main() -> None:
    classifier = TFIDFIntentClassifier(INTENT_EXAMPLES)
    agents = [BillingAgent(), TechSupportAgent(), AccountAgent(), FallbackAgent()]
    orchestrator = Orchestrator(agents, classifier)

    session = Session(session_id="demo-1", account_context={"account_id": "acct_1003"})

    turns = [
        "I have an issue",  # vague -> triggers contextual follow-up
        "why is my bill so high",
        "what plan am I on",
        "asdkjf random gibberish",  # low confidence -> fallback
    ]

    for utterance in turns:
        result = orchestrator.handle_turn(utterance, session)
        print(f"user: {utterance}")
        print(
            f"  -> routed_to={result.routed_intent} "
            f"confidence={result.routing_confidence:.2f} "
            f"guardrail_passed={result.guardrail_passed}"
        )
        print(f"  agent[{result.response.agent_name}]: {result.response.text}")
        if result.response.tool_calls:
            print(f"  tool_calls: {result.response.tool_calls}")
        print()


if __name__ == "__main__":
    main()
