"""Real LLM components (optional; needs ``pip install -e .[anthropic]`` and an API key).

* ``LLMExtractor`` — drop-in replacement for ``SimulatedExtractor``: asks Claude
  whether a user turn states a durable fact *about the user* and returns
  ``(attr, value)`` or ``None``.
* ``llm_answer`` — end-to-end reader: answers a question from a memory's context,
  or abstains.

The published results use the simulated extractor and oracle/gullible readers so
they are reproducible offline; these adapters let you measure where a real model
sits between those bounds.
"""

from __future__ import annotations

import json

from .memory import Snippet
from .synth import SCHEMA, Question, Turn

MODEL = "claude-opus-5-5"

EXTRACT_PROMPT = """You maintain long-term memory about a user. Attributes: {attrs}.
Does the user's message below state a durable fact about THE USER THEMSELVES for one
of those attributes? Mentions of other people (friends, family, colleagues) do not count.
Reply with JSON only: {{"attr": <attribute or null>, "value": <value or null>}}.

Message: {text}"""

ANSWER_PROMPT = """Answer the user's question using only the memory context below.
If the context does not contain the answer, reply exactly: UNKNOWN.
Reply with the answer value only.

Memory context:
{ctx}

Question: {q}"""


class LLMExtractor:
    def __init__(self, client=None, model: str = MODEL):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client, self.model = client, model

    def __call__(self, t: Turn, user_id: int) -> tuple[str, str] | None:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACT_PROMPT.format(attrs=", ".join(SCHEMA), text=t.text),
                }
            ],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        try:
            d = json.loads(text[text.index("{") : text.rindex("}") + 1])
        except ValueError:
            return None
        if d.get("attr") in SCHEMA and d.get("value"):
            return d["attr"], str(d["value"])
        return None


def llm_answer(client, q: Question, ctx: list[Snippet], model: str = MODEL) -> str | None:
    msg = client.messages.create(
        model=model,
        max_tokens=50,
        messages=[
            {
                "role": "user",
                "content": ANSWER_PROMPT.format(ctx="\n".join(s.text for s in ctx), q=q.text),
            }
        ],
    )
    out = "".join(b.text for b in msg.content if b.type == "text").strip()
    return None if out.upper().startswith("UNKNOWN") else out


def grade(pred: str | None, gold: str | None) -> bool:
    if gold is None or pred is None:
        return pred is None and gold is None
    return gold.lower() in pred.lower()
