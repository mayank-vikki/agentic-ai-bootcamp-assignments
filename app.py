"""
Day 2 Assignment — Routing with LangGraph (Tier-Based Support Flow)

Bootcamp: Agentic AI Enterprise Mastery (Manifold AI)
Model:    DeepSeek Chat (via OpenAI-compatible API)

Demonstrates:
  1. Typed conversation state with SupportState (TypedDict)
  2. Conditional routing based on user tier (VIP vs standard)
  3. Explicit, auditable graph structure using LangGraph
"""

import operator
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from utils.models import get_model
from utils.logger import logged_invoke

load_dotenv()


# ── State Definition ─────────────────────────────────────────────────────────

class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    should_escalate: bool
    issue_type: str
    user_tier: str  # "vip" or "standard"


# ── Initialize LLM ──────────────────────────────────────────────────────────

llm = get_model()  # DeepSeek → OpenAI fallback
MODEL = getattr(llm, "_bootcamp_model_name", "deepseek")


# ── Routing Function ─────────────────────────────────────────────────────────

def route_by_tier(state: SupportState) -> str:
    """Route based on user tier."""
    if state.get("user_tier") == "vip":
        return "vip_path"
    return "standard_path"


# ── Node Functions ───────────────────────────────────────────────────────────

def check_user_tier_node(state: SupportState):
    """Decide if user is VIP or standard based on message content."""
    first_message = state["messages"][0].content.lower()
    if "vip" in first_message or "premium" in first_message:
        return {"user_tier": "vip"}
    return {"user_tier": "standard"}


def vip_agent_node(state: SupportState):
    """VIP path: priority handling with LLM response, no escalation."""
    response = logged_invoke(
        llm,
        [
            SystemMessage(
                content="You are a premium support agent. The customer is a VIP. "
                "Be concise, professional, and prioritize their request. "
                "Respond in 2-3 sentences."
            ),
            *state["messages"],
        ],
        model_name=MODEL,
    )
    return {
        "messages": [response],
        "should_escalate": False,
        "issue_type": "vip_support",
    }


def standard_agent_node(state: SupportState):
    """Standard path: general handling, may escalate to human agent."""
    response = logged_invoke(
        llm,
        [
            SystemMessage(
                content="You are a support agent handling a standard-tier request. "
                "Be helpful but note that complex issues may need escalation. "
                "Respond in 2-3 sentences."
            ),
            *state["messages"],
        ],
        model_name=MODEL,
    )
    return {
        "messages": [response],
        "should_escalate": True,
        "issue_type": "standard_support",
    }


# ── Graph Construction ───────────────────────────────────────────────────────

def build_graph():
    """Build and compile the tier-based routing graph."""
    workflow = StateGraph(SupportState)

    workflow.add_node("check_tier", check_user_tier_node)
    workflow.add_node("vip_agent", vip_agent_node)
    workflow.add_node("standard_agent", standard_agent_node)

    workflow.set_entry_point("check_tier")

    workflow.add_conditional_edges(
        "check_tier",
        route_by_tier,
        {
            "vip_path": "vip_agent",
            "standard_path": "standard_agent",
        },
    )

    workflow.add_edge("vip_agent", END)
    workflow.add_edge("standard_agent", END)

    return workflow.compile()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    graph = build_graph()

    # ── Run 1: VIP customer ──────────────────────────────────────────────
    print("=" * 70)
    print("RUN 1: VIP Customer")
    print("=" * 70)

    vip_result = graph.invoke({
        "messages": [HumanMessage(content="I'm a VIP customer, please check my order")],
        "should_escalate": False,
        "issue_type": "",
        "user_tier": "",
    })

    print(f"\n  Tier:     {vip_result.get('user_tier')}")
    print(f"  Escalate: {vip_result.get('should_escalate')}")
    print(f"  Type:     {vip_result.get('issue_type')}")
    print(f"  Response: {vip_result['messages'][-1].content[:200]}")

    # ── Run 2: Standard customer ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RUN 2: Standard Customer")
    print("=" * 70)

    standard_result = graph.invoke({
        "messages": [HumanMessage(content="Check my order status")],
        "should_escalate": False,
        "issue_type": "",
        "user_tier": "",
    })

    print(f"\n  Tier:     {standard_result.get('user_tier')}")
    print(f"  Escalate: {standard_result.get('should_escalate')}")
    print(f"  Type:     {standard_result.get('issue_type')}")
    print(f"  Response: {standard_result['messages'][-1].content[:200]}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ROUTING SUMMARY")
    print("=" * 70)
    print(f"  VIP result:      tier={vip_result['user_tier']}, escalate={vip_result['should_escalate']}")
    print(f"  Standard result: tier={standard_result['user_tier']}, escalate={standard_result['should_escalate']}")


if __name__ == "__main__":
    main()


"""
Reflection:

1. Why use typed state (SupportState) instead of a plain dict?
   TypedDict provides compile-time and IDE-level visibility into what fields
   the graph operates on. Every node reads and writes a known contract. When
   the graph grows (adding escalation nodes, feedback loops), typed state
   prevents silent key mismatches that plain dicts allow.

2. Why is explicit routing (route_by_tier) better than embedding logic in nodes?
   The routing function is a single, testable unit. You can write a unit test
   for route_by_tier without invoking the graph. In production, routing decisions
   are auditable — you can log which path was taken without parsing node output.
   This separation of routing from processing is the core LangGraph pattern.

3. What would you change for production?
   - Replace the keyword-based tier check with a database lookup (CRM integration)
   - Add an escalation node that hands off to a human agent queue
   - Add observability: log which path was taken, latency per node, and LLM cost
   - Add a feedback edge: if the VIP agent can't resolve, route to escalation
   - Use LangGraph's persistence (checkpointers) for multi-turn conversations
"""
