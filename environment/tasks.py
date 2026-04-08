"""
Task definitions and scenario data for all three difficulty levels.

Task 1 – Easy   : Issue Classification (single step)
Task 2 – Medium : Response Generation (single step, rule-graded)
Task 3 – Hard   : Full Resolution (multi-step, progressive rewards)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class ClassifyScenario:
    query: str
    correct_category: str


@dataclass
class RespondScenario:
    query: str
    category: str
    required_keywords: List[str]          # at least one must appear
    politeness_words: List[str] = field(default_factory=list)
    context_hint: str = ""


@dataclass
class ResolveScenario:
    query: str
    category: str
    context_hint: str = ""
    clarification_keywords: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# Politeness vocabulary (shared)
# ──────────────────────────────────────────────

POLITENESS_WORDS: List[str] = [
    "sorry", "apologize", "apologies", "thank you", "thanks",
    "understand", "unfortunately", "certainly", "happy to help",
    "glad to", "please", "regret", "inconvenience", "assist",
]


# ──────────────────────────────────────────────
# Task 1 – Easy: Classification
# ──────────────────────────────────────────────

CLASSIFY_SCENARIOS: List[ClassifyScenario] = [
    ClassifyScenario("I was charged twice for my last order.",        "billing"),
    ClassifyScenario("My package hasn't arrived in two weeks.",       "shipping"),
    ClassifyScenario("I want to return this product and get a refund.", "refund"),
    ClassifyScenario("The app keeps crashing during checkout.",       "technical"),
    ClassifyScenario("What are your store opening hours?",            "general"),
    ClassifyScenario("I received the wrong item in my delivery.",     "shipping"),
    ClassifyScenario("My credit card was charged but the order failed.", "billing"),
    ClassifyScenario("How do I reset my account password?",           "technical"),
    ClassifyScenario("Can I change my delivery address?",             "shipping"),
    ClassifyScenario("I need an invoice for my purchase.",            "billing"),
    ClassifyScenario("The product stopped working after two days.",   "refund"),
    ClassifyScenario("Do you offer student discounts?",               "general"),
]

VALID_CATEGORIES: List[str] = ["billing", "shipping", "refund", "technical", "general"]


# ──────────────────────────────────────────────
# Task 2 – Medium: Response Generation
# ──────────────────────────────────────────────

RESPOND_SCENARIOS: List[RespondScenario] = [
    RespondScenario(
        query="My order has been delayed for 5 days. I'm very frustrated!",
        category="shipping",
        required_keywords=["order", "delay", "delayed", "shipment", "delivery"],
        context_hint="Respond empathetically about the shipping delay.",
    ),
    RespondScenario(
        query="I want a refund for my broken product. It arrived damaged.",
        category="refund",
        required_keywords=["refund", "return", "damage", "broken", "replacement"],
        context_hint="Acknowledge the damage and explain the refund process.",
    ),
    RespondScenario(
        query="I was charged twice and need this fixed immediately!",
        category="billing",
        required_keywords=["charge", "charged", "billing", "payment", "duplicate"],
        context_hint="Address the double charge urgently.",
    ),
    RespondScenario(
        query="Your website is down and I can't complete my purchase.",
        category="technical",
        required_keywords=["website", "technical", "issue", "access", "purchase", "problem"],
        context_hint="Acknowledge the technical issue and offer alternatives.",
    ),
    RespondScenario(
        query="I never received a confirmation email for my order.",
        category="shipping",
        required_keywords=["email", "confirmation", "order", "receipt"],
        context_hint="Help the customer locate their order confirmation.",
    ),
    RespondScenario(
        query="The tracking link for my package shows no updates for a week.",
        category="shipping",
        required_keywords=["tracking", "package", "update", "shipment"],
        context_hint="Investigate the tracking issue and reassure the customer.",
    ),
]


# ──────────────────────────────────────────────
# Task 3 – Hard: Full Resolution
# ──────────────────────────────────────────────

RESOLVE_SCENARIOS: List[ResolveScenario] = [
    ResolveScenario(
        query="I got charged twice for my order. This is completely unacceptable!",
        category="billing",
        context_hint="Guide agent through: identify issue → ask for order/payment details → resolve the double charge.",
        clarification_keywords=["order", "id", "number", "transaction", "payment", "date", "amount"],
    ),
    ResolveScenario(
        query="My package hasn't arrived. It's been almost three weeks since I ordered!",
        category="shipping",
        context_hint="Guide agent through: identify issue → ask for order ID or address → initiate investigation.",
        clarification_keywords=["order", "id", "number", "address", "tracking", "date"],
    ),
    ResolveScenario(
        query="I received a completely wrong item and need this sorted out now.",
        category="shipping",
        context_hint="Guide agent through: identify wrong item → ask for order ID and photo/details → arrange exchange or refund.",
        clarification_keywords=["order", "id", "number", "item", "photo", "received"],
    ),
    ResolveScenario(
        query="My account was somehow charged even though I cancelled the subscription.",
        category="billing",
        context_hint="Guide agent through: confirm cancellation issue → ask for account details → process refund.",
        clarification_keywords=["account", "email", "subscription", "date", "cancel", "charge"],
    ),
]


# ──────────────────────────────────────────────
# Task metadata (used by openenv.yaml and API)
# ──────────────────────────────────────────────

TASK_META = {
    "classify": {
        "name": "Issue Classification",
        "difficulty": "easy",
        "task_type": "easy",
        "max_steps": 1,
        "description": (
            "The agent receives a customer query and must classify it into one of "
            "five categories: billing, shipping, refund, technical, or general."
        ),
    },
    "respond": {
        "name": "Response Generation",
        "difficulty": "medium",
        "task_type": "medium",
        "max_steps": 1,
        "description": (
            "The agent receives a customer complaint and must compose a relevant, "
            "polite response that addresses the specific issue."
        ),
    },
    "resolve": {
        "name": "Full Issue Resolution",
        "difficulty": "hard",
        "task_type": "hard",
        "max_steps": 5,
        "description": (
            "The agent must fully resolve a customer issue over multiple turns: "
            "identify the problem, ask for clarification, and provide a concrete solution."
        ),
    },
}