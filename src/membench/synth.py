"""Synthetic multi-session user histories with knowledge updates and hard distractors.

Each simulated user talks to an assistant over many sessions. Some turns state a
personal fact (``city = Pune``). Later sessions may **update** it (moved to
Berlin). Most turns are filler, and some are **hard distractors** that mention a
candidate value about *someone else* ("my sister just moved to Berlin").

Questions are phrased differently from statements (lexical gap), in four types:

* ``current``  — the latest value of an attribute stated once;
* ``updated``  — the latest value of an attribute that changed (stale values exist);
* ``previous`` — the value *before* the latest one (temporal reasoning);
* ``absent``   — never stated; the right answer is to abstain.

Every turn carries ground-truth metadata (``fact``) used only for grading and by
the oracle reader — never by the memory systems themselves.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# attribute -> (values, user statement templates, distractor templates, question paraphrases)
SCHEMA: dict[str, dict] = {
    "city": {
        "values": ["Pune", "Berlin", "Austin", "Lisbon", "Nairobi", "Osaka", "Toronto", "Leeds"],
        "state": ["I live in {v} these days.", "Just finished moving to {v}!", "Home is {v} now."],
        "distract": ["My sister just moved to {v}.", "We might visit {v} next spring."],
        "ask": ["Which city am I based in?", "Where is my home right now?"],
    },
    "employer": {
        "values": [
            "Acme Corp",
            "Globex",
            "Initech",
            "Umbrella Labs",
            "Stark Systems",
            "Wayne Data",
        ],
        "state": ["I work at {v}.", "Started my new job at {v} this week.", "{v} is my employer."],
        "distract": ["A recruiter from {v} pinged me.", "My friend got an offer from {v}."],
        "ask": ["Which company do I work for?", "Who pays my salary?"],
    },
    "pet": {
        "values": ["Biscuit", "Luna", "Pepper", "Mochi", "Rex", "Tofu"],
        "state": ["My dog is called {v}.", "We named our new puppy {v}."],
        "distract": ["My neighbour's cat is named {v}.", "{v} is such a cute name for a pet."],
        "ask": ["What's my dog's name?", "What did we call our puppy?"],
    },
    "diet": {
        "values": ["vegetarian", "vegan", "pescatarian", "keto", "gluten-free"],
        "state": ["I eat {v} now.", "Heads up, I've gone {v}."],
        "distract": ["My roommate is trying {v} this month.", "Is a {v} diet healthy?"],
        "ask": ["What diet do I follow?", "Any food restrictions I have?"],
    },
    "language": {
        "values": ["Rust", "Go", "Python", "TypeScript", "Kotlin", "Elixir"],
        "state": ["{v} is my main programming language.", "I mostly code in {v} at work."],
        "distract": ["Our intern keeps asking about {v}.", "Saw a talk on {v} yesterday."],
        "ask": ["Which programming language do I use most?", "What do I code in day to day?"],
    },
    "car": {
        "values": ["Tesla Model 3", "Honda Civic", "Toyota Prius", "Mazda 3", "VW Golf"],
        "state": ["I drive a {v}.", "Picked up my {v} from the dealer today."],
        "distract": ["My dad is thinking of buying a {v}.", "A {v} cut me off on the highway."],
        "ask": ["What car do I own?", "What vehicle do I drive?"],
    },
    "manager": {
        "values": ["Priya", "Marcus", "Elena", "Tom", "Aisha", "Kenji"],
        "state": ["My manager is {v}.", "I report to {v} now."],
        "distract": ["{v} from finance sent a long email.", "Had lunch with {v} from sales."],
        "ask": ["Who is my boss?", "Who do I report to?"],
    },
    "gym_day": {
        "values": ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday"],
        "state": ["I hit the gym every {v}.", "Leg day for me is {v}."],
        "distract": ["The gym is packed on {v}.", "Our team sync moved to {v}."],
        "ask": ["Which day do I work out?", "When is my weekly training session?"],
    },
    "allergy": {
        "values": ["peanuts", "shellfish", "pollen", "penicillin", "dairy"],
        "state": ["I'm allergic to {v}.", "Turns out I have a {v} allergy."],
        "distract": ["My nephew reacts badly to {v}.", "This menu lists {v} everywhere."],
        "ask": ["What am I allergic to?", "Anything that triggers my allergies?"],
    },
    "project": {
        "values": ["Falcon", "Nimbus", "Atlas", "Hermes", "Orion", "Kestrel"],
        "state": ["I'm leading project {v}.", "My team's project is codenamed {v}."],
        "distract": ["Project {v} got cancelled in another org.", "Read a blog about {v}."],
        "ask": ["What's the codename of the project I lead?", "Which project am I running?"],
    },
}

# Hand-written attribute descriptions for query routing (independent of the templates).
DESCRIPTIONS: dict[str, str] = {
    "city": "city town home location where the user lives based residence",
    "employer": "company employer job workplace organisation salary works for",
    "pet": "pet dog cat puppy animal name",
    "diet": "diet food eating restrictions vegetarian vegan meals",
    "language": "programming language code coding software day to day",
    "car": "car vehicle drive owns automobile",
    "manager": "manager boss supervisor reports to lead",
    "gym_day": "gym workout exercise training day weekly",
    "allergy": "allergy allergic reaction triggers",
    "project": "project codename lead running initiative",
}

FILLER = [
    "Can you help me draft an email to the team?",
    "What's a good recipe for a quick dinner?",
    "Summarise the main points of the article I pasted.",
    "How do I fix a merge conflict in git?",
    "Any tips for sleeping better?",
    "Explain the difference between TCP and UDP.",
    "Recommend a sci-fi book please.",
    "What's the weather usually like in autumn?",
    "Help me plan a birthday party for ten people.",
    "Translate 'good morning' into Spanish.",
    "Write a haiku about coffee.",
    "How long should I boil an egg?",
    "What are some good stretching exercises?",
    "Give me a packing list for a weekend trip.",
    "How do vaccines work, roughly?",
    "Suggest names for a team offsite.",
]
ASSISTANT = ["Sure, here you go.", "Happy to help!", "Got it.", "Here's a suggestion.", "Noted."]


@dataclass
class Turn:
    session: int
    day: int
    role: str  # "user" | "assistant"
    text: str
    fact: tuple[str, str] | None = None  # ground truth: user states attr = value
    mentions: tuple[str, str] | None = None  # distractor: value mentioned, not about user


@dataclass
class Question:
    qtype: str  # current | updated | previous | absent
    attr: str
    text: str
    answer: str | None
    day: int  # asked after the last session
    evidence_day: int | None  # day of the gold statement


@dataclass
class History:
    user_id: int
    turns: list[Turn] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)


def make_history(
    user_id: int,
    n_sessions: int = 20,
    turns_per_session: int = 12,
    n_facts: int = 7,
    n_updates: int = 4,
    distractor_rate: float = 0.15,
    seed: int = 0,
) -> History:
    """One user's history. ``n_facts`` attributes get stated; ``n_updates`` of them change."""
    rng = random.Random(seed * 100_003 + user_id)
    attrs = list(SCHEMA)
    rng.shuffle(attrs)
    stated, absent = attrs[:n_facts], attrs[n_facts:]
    updated = set(stated[:n_updates])

    # Plan statement events: (session, attr, value)
    events: list[tuple[int, str, str]] = []
    timeline: dict[str, list[tuple[int, str]]] = {}
    for a in stated:
        vals = rng.sample(SCHEMA[a]["values"], 3)
        n_vals = (2 + (rng.random() < 0.4)) if a in updated else 1
        sessions = sorted(rng.sample(range(n_sessions), n_vals))
        for s, v in zip(sessions, vals[:n_vals], strict=True):
            events.append((s, a, v))
    by_session: dict[int, list[tuple[str, str]]] = {}
    for s, a, v in events:
        by_session.setdefault(s, []).append((a, v))

    h = History(user_id)
    for s in range(n_sessions):
        day = s * 3 + rng.randint(0, 2)
        slots = ["filler"] * turns_per_session
        stmt = by_session.get(s, [])
        positions = rng.sample(range(turns_per_session), len(stmt))
        for p, (a, v) in zip(positions, stmt, strict=True):
            slots[p] = (a, v)
        for i, slot in enumerate(slots):
            if isinstance(slot, tuple):
                a, v = slot
                text = rng.choice(SCHEMA[a]["state"]).format(v=v)
                h.turns.append(Turn(s, day, "user", text, fact=(a, v)))
                timeline.setdefault(a, []).append((day, v))
            elif rng.random() < distractor_rate:
                a = rng.choice(attrs)
                v = rng.choice(SCHEMA[a]["values"])
                text = rng.choice(SCHEMA[a]["distract"]).format(v=v)
                h.turns.append(Turn(s, day, "user", text, mentions=(a, v)))
            else:
                h.turns.append(Turn(s, day, "user", rng.choice(FILLER)))
            if i % 2 == 1:
                h.turns.append(Turn(s, day, "assistant", rng.choice(ASSISTANT)))

    q_day = h.turns[-1].day + 1
    for a, tl in timeline.items():
        ask = rng.choice(SCHEMA[a]["ask"])
        last_day, last_v = tl[-1]
        qtype = "updated" if len(tl) > 1 else "current"
        h.questions.append(Question(qtype, a, ask, last_v, q_day, last_day))
        if len(tl) > 1:
            prev_day, prev_v = tl[-2]
            q = f"Before my latest change, what was the answer to: {ask[0].lower()}{ask[1:]}"
            h.questions.append(Question("previous", a, q, prev_v, q_day, prev_day))
    for a in absent:
        h.questions.append(Question("absent", a, rng.choice(SCHEMA[a]["ask"]), None, q_day, None))
    return h


def make_dataset(n_users: int = 50, seed: int = 0, **kw) -> list[History]:
    return [make_history(u, seed=seed, **kw) for u in range(n_users)]
