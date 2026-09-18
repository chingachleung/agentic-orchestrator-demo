# agentic-orchestrator-demo

[![tests](https://github.com/chingachleung/agentic-orchestrator-demo/actions/workflows/tests.yml/badge.svg)](https://github.com/chingachleung/agentic-orchestrator-demo/actions/workflows/tests.yml)

A small, runnable multi-agent orchestration system: an orchestrator classifies
each incoming message, routes it to the right specialist agent, lets that
agent call backend tools to ground its answer in real state, and passes every
response through an output guardrail before it reaches the user.

This is an original, from-scratch demo — not production code — built to show
the same architectural pattern (classify → route → tool-call → guardrail →
respond, with shared session state across hops) used in customer-support
agent systems I've built professionally.

## Why this design

- **Pluggable routing brain.** The orchestrator depends only on an
  `IntentClassifier` interface, not a specific model. It ships with a
  zero-dependency TF-IDF classifier so the demo runs with no API key and no
  model download. A second, fully-written `LLMIntentClassifier` implements
  the same interface against the Anthropic API for higher-accuracy routing —
  it's real, correct integration code, just inactive until you supply
  `ANTHROPIC_API_KEY` (see below). Swapping one for the other requires no
  change to `Orchestrator` or any agent.
- **Contextual, not generic, clarifying questions.** When a request is vague
  (`"I have an issue"`), agents don't just ask "can you clarify?" — they pull
  the relevant account context via a tool call first, then ask a targeted
  follow-up (e.g. flagging a past-due balance if that's likely relevant).
- **Tool permissions are explicit.** Each agent declares which tools it's
  allowed to call; calling anything else raises `PermissionError`.
- **Guardrail on every response.** Before anything reaches the "user," it
  passes through `output_guardrail`, which blocks responses that would echo
  back sensitive identifiers. A blocked response is swapped for a safe
  deterministic handoff message, not silently sent.
- **Shared session state.** One `Session` object flows through the whole
  turn so agents aren't re-deriving account context or conversation history
  on every hop.

## What's real vs. mocked

| Component | Status |
|---|---|
| Orchestration / routing / guardrail logic | Real, fully functional |
| `TFIDFIntentClassifier` | Real, runs locally, no dependencies beyond scikit-learn |
| `LLMIntentClassifier` | Real integration code, **inactive without an API key** |
| Backend account/billing lookups | Mocked in-memory data — no real backend exists here |

## Run it

```bash
pip install -r requirements.txt
python examples/run_demo.py
```

No API key needed for the default run.

### Example output

```
user: I have an issue
  -> routed_to=tech_support confidence=0.58 guardrail_passed=True
  agent[tech_support_agent]: I see you're on Basic Internet — is the issue with your internet connection or a specific device?
  tool_calls: ['lookup_account']

user: why is my bill so high
  -> routed_to=billing_inquiry confidence=1.00 guardrail_passed=True
  agent[billing_agent]: Your current balance due is $118.20.
  tool_calls: ['check_billing_status']

user: what plan am I on
  -> routed_to=account_management confidence=1.00 guardrail_passed=True
  agent[account_agent]: Your account is on the Basic Internet plan and is currently past_due.
  tool_calls: ['lookup_account']

user: asdkjf random gibberish
  -> routed_to=unknown confidence=0.00 guardrail_passed=True
  agent[fallback_agent]: I want to make sure I route you correctly — could you rephrase that?
```

Note the third turn: it's routed to `account_management`, and that agent's
tool call to `lookup_account` is what surfaces the past-due status baked
into the mocked account data — a concrete example of the "ground the answer
in real state via a tool call" pattern, not a canned string.

To try the LLM-backed classifier instead:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
```

```python
from src import Orchestrator, LLMIntentClassifier, BillingAgent, TechSupportAgent, AccountAgent, FallbackAgent

classifier = LLMIntentClassifier()
orchestrator = Orchestrator(
    [BillingAgent(), TechSupportAgent(), AccountAgent(), FallbackAgent()],
    classifier,
)
```

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

## Layout

```
src/
  orchestrator.py       # classify -> route -> guardrail -> respond
  agents.py              # specialist agents + contextual follow-up logic
  intent_classifier.py   # pluggable routing backend (TF-IDF default, LLM hook)
  tools.py                # mocked backend calls + guardrail tool
  session.py              # shared conversation/account state
examples/run_demo.py      # end-to-end runnable demo
tests/                     # pytest suite
```
