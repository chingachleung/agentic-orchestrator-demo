"""Orchestrator: routes a turn to the right specialist agent, enforces the
output guardrail on the way out, and keeps session state flowing between
hops.

This is the piece a production system's "steering agent" plays: classify →
dispatch → guardrail → respond, with everything routed through one shared
Session so downstream agents don't need to re-derive context.
"""
from __future__ import annotations

from dataclasses import dataclass

from .agents import AgentResponse, BaseAgent
from .intent_classifier import IntentClassifier
from .session import Session
from .tools import TOOL_REGISTRY


@dataclass
class OrchestratorResult:
    response: AgentResponse
    routed_intent: str
    routing_confidence: float
    guardrail_passed: bool
    guardrail_reason: str | None = None


class Orchestrator:
    def __init__(self, agents: list[BaseAgent], classifier: IntentClassifier, confidence_floor: float = 0.15):
        self.agents_by_intent: dict[str, BaseAgent] = {
            intent: agent for agent in agents for intent in agent.owned_intents
        }
        self.classifier = classifier
        self.confidence_floor = confidence_floor
        self.fallback = next((a for a in agents if a.name == "fallback_agent"), None)

    def handle_turn(self, utterance: str, session: Session) -> OrchestratorResult:
        session.add_turn("user", utterance)

        candidate_intents = list(self.agents_by_intent.keys())
        result = self.classifier.classify(utterance, candidate_intents)

        if result.confidence < self.confidence_floor or result.intent not in self.agents_by_intent:
            agent = self.fallback
            routed_intent = "unknown"
        else:
            agent = self.agents_by_intent[result.intent]
            routed_intent = result.intent

        response = agent.handle(utterance, session)

        guardrail = TOOL_REGISTRY["output_guardrail"](text=response.text)
        if not guardrail["allowed"]:
            # Deterministic flow handoff: guardrail failure never reaches the
            # user verbatim — swap in a safe, generic response instead.
            response = AgentResponse(agent.name, "Let me connect you with someone who can help with that.")

        session.add_turn("agent", response.text, agent_name=agent.name)

        return OrchestratorResult(
            response=response,
            routed_intent=routed_intent,
            routing_confidence=result.confidence,
            guardrail_passed=guardrail["allowed"],
            guardrail_reason=guardrail["reason"],
        )
