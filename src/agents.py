"""Specialist agents the orchestrator can route to.

Each agent declares the intents it owns and the tools it's allowed to call.
Deliberately simple response logic (mostly templated) — the point of this
demo is the *routing, tool-calling, and guardrail architecture*, not a fancy
response generator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .session import Session
from .tools import TOOL_REGISTRY, Tool


VAGUE_MARKERS = ("something", "issue", "problem", "help", "not working", "wrong")


@dataclass
class AgentResponse:
    agent_name: str
    text: str
    tool_calls: list[str] = field(default_factory=list)
    follow_up_question: str | None = None


class BaseAgent:
    name: str = "base"
    owned_intents: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()

    def tool(self, tool_name: str) -> Tool:
        if tool_name not in self.allowed_tools:
            raise PermissionError(f"{self.name} is not permitted to call {tool_name!r}")
        return TOOL_REGISTRY[tool_name]

    def handle(self, utterance: str, session: Session) -> AgentResponse:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def is_vague(utterance: str) -> bool:
        lowered = utterance.lower()
        return any(marker in lowered for marker in VAGUE_MARKERS) and len(utterance.split()) < 8


class BillingAgent(BaseAgent):
    name = "billing_agent"
    owned_intents = ("billing_inquiry",)
    allowed_tools = ("check_billing_status", "output_guardrail")

    def handle(self, utterance: str, session: Session) -> AgentResponse:
        account_id = session.account_context.get("account_id")

        # Contextual-follow-up pattern: on a vague request, pull account
        # context first and use it to ask a targeted question rather than a
        # generic clarifier.
        if self.is_vague(utterance) and account_id:
            billing = self.tool("check_billing_status")(account_id=account_id)
            tool_calls = ["check_billing_status"]
            if billing.get("past_due"):
                question = (
                    f"I can see your account has a past-due balance of "
                    f"${billing['balance_due']:.2f} — is that what you're asking about?"
                )
            else:
                question = "Your account balance looks current — could you tell me more about what you're seeing?"
            return AgentResponse(self.name, question, tool_calls, follow_up_question=question)

        if account_id:
            billing = self.tool("check_billing_status")(account_id=account_id)
            text = f"Your current balance due is ${billing.get('balance_due', 0):.2f}."
            return AgentResponse(self.name, text, ["check_billing_status"])

        return AgentResponse(self.name, "Could you share your account ID so I can check your balance?")


class TechSupportAgent(BaseAgent):
    name = "tech_support_agent"
    owned_intents = ("tech_support",)
    allowed_tools = ("lookup_account", "output_guardrail")

    def handle(self, utterance: str, session: Session) -> AgentResponse:
        account_id = session.account_context.get("account_id")
        if self.is_vague(utterance) and account_id:
            account = self.tool("lookup_account")(account_id=account_id)
            plan = account.get("plan", "your plan")
            question = f"I see you're on {plan} — is the issue with your internet connection or a specific device?"
            return AgentResponse(self.name, question, ["lookup_account"], follow_up_question=question)
        return AgentResponse(self.name, "Let's troubleshoot — is the router showing any warning lights?")


class AccountAgent(BaseAgent):
    name = "account_agent"
    owned_intents = ("account_management",)
    allowed_tools = ("lookup_account", "output_guardrail")

    def handle(self, utterance: str, session: Session) -> AgentResponse:
        account_id = session.account_context.get("account_id")
        if not account_id:
            return AgentResponse(self.name, "What's your account ID?")
        account = self.tool("lookup_account")(account_id=account_id)
        text = f"Your account is on the {account.get('plan')} plan and is currently {account.get('status')}."
        return AgentResponse(self.name, text, ["lookup_account"])


class FallbackAgent(BaseAgent):
    name = "fallback_agent"
    owned_intents = ("unknown",)
    allowed_tools = ()

    def handle(self, utterance: str, session: Session) -> AgentResponse:
        return AgentResponse(self.name, "I want to make sure I route you correctly — could you rephrase that?")
