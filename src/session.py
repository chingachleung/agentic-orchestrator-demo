"""Session state shared across an orchestrator and its sub-agents.

Mirrors a common production pattern: rather than re-fetching context on every
agent hop, the orchestrator forwards a single Session object so downstream
agents can read prior turns, account context, and per-turn scratch data
without extra round trips.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    speaker: str  # "user" | "agent"
    text: str
    agent_name: str | None = None


@dataclass
class Session:
    session_id: str
    history: list[Turn] = field(default_factory=list)
    account_context: dict[str, Any] = field(default_factory=dict)
    scratch: dict[str, Any] = field(default_factory=dict)

    def add_turn(self, speaker: str, text: str, agent_name: str | None = None) -> None:
        self.history.append(Turn(speaker=speaker, text=text, agent_name=agent_name))

    def recent_text(self, n: int = 6) -> str:
        """Compact recent history for passing to a classifier/tool — a light
        stand-in for the conversation-history compaction a production system
        would need once sessions run long."""
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.history[-n:])
