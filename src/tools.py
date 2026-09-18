"""LLM-callable tools: specialist-facing backend calls, plus an output
guardrail every agent response passes through before reaching the user.

Backend calls here are mocked with a small in-memory "database" — there's no
real account system behind this demo — but the calling contract (name,
description, args, structured return) mirrors how these would be registered
as callable tools with a real agent/LLM framework (e.g. ADK, function
calling with an LLM provider).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


# ---- mock backend "database" -------------------------------------------------

_ACCOUNTS: dict[str, dict[str, Any]] = {
    "acct_1001": {"plan": "Fiber 500", "balance_due": 42.50, "status": "active"},
    "acct_1002": {"plan": "Fiber 1000", "balance_due": 0.0, "status": "active"},
    "acct_1003": {"plan": "Basic Internet", "balance_due": 118.20, "status": "past_due"},
}


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)


def lookup_account(account_id: str) -> dict[str, Any]:
    """Mocked backend call a specialist agent uses to ground its answer in
    real account state instead of guessing."""
    return _ACCOUNTS.get(account_id, {"error": "account_not_found"})


def check_billing_status(account_id: str) -> dict[str, Any]:
    account = _ACCOUNTS.get(account_id)
    if account is None:
        return {"error": "account_not_found"}
    return {
        "balance_due": account["balance_due"],
        "past_due": account["status"] == "past_due",
    }


_BLOCKED_PATTERNS = [
    re.compile(r"\bssn\b", re.I),
    re.compile(r"\bsocial security\b", re.I),
    re.compile(r"\bcredit card number\b", re.I),
]


def output_guardrail(text: str) -> dict[str, Any]:
    """Deterministic output guardrail: blocks responses that would echo back
    sensitive account identifiers, mirroring a production guardrail tool
    gating high-risk / compliance-sensitive responses before they reach a
    customer."""
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(text):
            return {"allowed": False, "reason": f"matched blocked pattern: {pattern.pattern}"}
    return {"allowed": True, "reason": None}


TOOL_REGISTRY: dict[str, Tool] = {
    "lookup_account": Tool("lookup_account", "Look up an account's plan/status", lookup_account),
    "check_billing_status": Tool(
        "check_billing_status", "Check whether an account has a past-due balance", check_billing_status
    ),
    "output_guardrail": Tool(
        "output_guardrail", "Validate an agent response before it reaches the user", output_guardrail
    ),
}
