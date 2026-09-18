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
    "billing_inquiry": ["why is my bill so high", "how much do I owe", "I want to pay my balance"],
    "tech_support": ["my internet is down", "the wifi keeps disconnecting", "I have an issue", "I have something"],
    "account_management": ["what plan am I on", "update my account info"],
}


def build_orchestrator() -> Orchestrator:
    classifier = TFIDFIntentClassifier(INTENT_EXAMPLES)
    agents = [BillingAgent(), TechSupportAgent(), AccountAgent(), FallbackAgent()]
    return Orchestrator(agents, classifier)


def test_routes_billing_question_to_billing_agent():
    orch = build_orchestrator()
    session = Session(session_id="t1", account_context={"account_id": "acct_1002"})
    result = orch.handle_turn("how much do I owe", session)
    assert result.response.agent_name == "billing_agent"
    assert result.routed_intent == "billing_inquiry"


def test_vague_request_triggers_contextual_follow_up():
    orch = build_orchestrator()
    session = Session(session_id="t2", account_context={"account_id": "acct_1003"})  # past_due
    result = orch.handle_turn("I have an issue", session)
    assert result.response.follow_up_question is not None
    assert "check_billing_status" in result.response.tool_calls or "lookup_account" in result.response.tool_calls


def test_low_confidence_falls_back():
    orch = build_orchestrator()
    session = Session(session_id="t3")
    result = orch.handle_turn("zzz qqq xyzzy", session)
    assert result.response.agent_name == "fallback_agent"


def test_guardrail_blocks_sensitive_echo():
    from src.tools import output_guardrail

    result = output_guardrail("Your SSN on file is 123-45-6789")
    assert result["allowed"] is False


def test_session_tracks_history():
    orch = build_orchestrator()
    session = Session(session_id="t4", account_context={"account_id": "acct_1001"})
    orch.handle_turn("what plan am I on", session)
    assert len(session.history) == 2  # user turn + agent turn
