"""
Day 1 Assignment — Context Failure → Context Fix (Production Mindset)

Bootcamp: Agentic AI Enterprise Mastery (Manifold AI)

Demonstrates:
  1. Why naive string-based LLM calls lose context (stateless behavior)
  2. How the Messages API preserves conversation state
  3. Production implications of ignoring message history
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ── Initialize the LLM ───────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4.1-nano")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Context Break Demonstration (Naive Invocation)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: Naive Invocation — Context Break Demo")
print("=" * 70)

# Two separate string-based llm.invoke() calls — context is lost between them
resp1 = llm.invoke("We are building an AI system for processing medical insurance claims.")
print(f"\n[Prompt 1] Response:\n{resp1.content}\n")

resp2 = llm.invoke("What are the main risks in this system?")
print(f"\n[Prompt 2] Response:\n{resp2.content}\n")

# WHY DOES THE SECOND CALL FAIL?
# Each llm.invoke() call is completely independent. The LLM has no memory
# of the first call. When we ask "What are the main risks in this system?",
# the model has no idea which "system" we're referring to — it never saw the
# medical insurance claims context.
#
# LLMs are STATELESS. Every API call starts with a blank slate.


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Context Fix Using Messages API
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 2: Messages API — Context Preserved")
print("=" * 70)

messages = [
    SystemMessage(
        content="You are a senior AI architect reviewing production systems."
    ),
    HumanMessage(
        content="We are building an AI system for processing medical insurance claims."
    ),
    HumanMessage(
        content="What are the main risks in this system?"
    ),
]

resp3 = llm.invoke(messages)
print(f"\n[Messages API] Response:\n{resp3.content}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Reflection Block (Mandatory)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Reflection:

1. Why did string-based invocation fail?
   Each call to llm.invoke(string) is a standalone request. The LLM receives
   only that single string with zero context from prior calls. When the second
   prompt says "this system", the model has no reference to "medical insurance
   claims" — it was never part of the input. This is the fundamental stateless
   nature of LLM APIs: no call remembers any previous call.

2. Why does message-based invocation work?
   The Messages API bundles the entire conversation — system instructions, prior
   human messages, and prior assistant responses — into a single structured list.
   When we send [SystemMessage, HumanMessage("insurance claims..."),
   HumanMessage("risks?")], the model sees the full thread in one shot. The
   "context" isn't stored server-side; it's explicitly re-sent by the client.

3. What would break in a production AI system if we ignore message history?
   - Multi-turn workflows (e.g., claim intake → validation → decision) would
     lose track of the case being processed.
   - The system would produce incoherent or contradictory responses across steps.
   - Users would need to repeat all context in every message, defeating the
     purpose of a conversational interface.
   - Audit trails would be meaningless because each response is disconnected.
   - In regulated domains (insurance, healthcare), this inconsistency could
     cause compliance violations and incorrect claim decisions.
"""
