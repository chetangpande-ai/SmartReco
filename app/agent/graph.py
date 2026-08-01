"""The recommendation graph.

                        ┌──────────┐
    START ─────────────▶│ analyze  │  profile -> query, filters, citable evidence
                        └────┬─────┘
                        ┌────▼─────┐
                        │   plan   │  is there enough signal to retrieve against?
                        └────┬─────┘
                  coldstart ◀┴─▶ agentic
              ┌───────────┐      ┌──────────┐
              │ coldstart │      │ retrieve │  vector ⊕ BM25 -> RRF -> filter -> MMR
              └─────┬─────┘      └────┬─────┘
                    │            ┌────▼─────┐
                    │            │  grade   │  LLM judge + rerank (fast model)
                    │            └────┬─────┘
                    │      score<0.6 ─┤─ ok
                    │       ┌─────────▼┐
                    │       │  refine  │ ──┐ reword + widen filters, max 2 loops
                    │       └──────────┘ ◀─┘ back to retrieve
                    │            ┌──────────┐
                    └───────────▶│ generate │  persuasive JSON (main model)
                                 └────┬─────┘
                                 ┌────▼─────┐
                                 │  verify  │  ids grounded? claims honest?
                                 └────┬─────┘
                        needs repair ─┤─ clean          (one repair attempt)
                                 ┌────▼─────┐
                                 │ finalize │ ─▶ END
                                 └──────────┘

Compiled once at import. It holds no per-request state — everything flows through
AgentState — so a single compiled graph is safe to share across requests.
"""

import logging

from langgraph.graph import END, START, StateGraph

from app.agent import nodes
from app.agent.state import AgentState

log = logging.getLogger(__name__)


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("analyze", nodes.analyze)
    g.add_node("plan", nodes.plan)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("coldstart", nodes.coldstart)
    g.add_node("grade", nodes.grade)
    g.add_node("refine", nodes.refine)
    g.add_node("generate", nodes.generate)
    g.add_node("verify", nodes.verify)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "analyze")
    g.add_edge("analyze", "plan")
    g.add_conditional_edges(
        "plan", nodes.route_after_plan, {"retrieve": "retrieve", "coldstart": "coldstart"}
    )
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade", nodes.route_after_grade, {"refine": "refine", "generate": "generate"}
    )
    g.add_edge("refine", "retrieve")
    g.add_edge("coldstart", "generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges(
        "verify", nodes.route_after_verify, {"generate": "generate", "finalize": "finalize"}
    )
    g.add_edge("finalize", END)

    return g.compile()


graph = build_graph()


def describe() -> dict:
    """Node and edge listing for the admin dashboard, read from the compiled graph
    rather than hand-maintained so it cannot drift from what actually runs."""
    compiled = graph.get_graph()
    return {
        "nodes": [n for n in compiled.nodes if n not in ("__start__", "__end__")],
        "edges": [
            {"from": e.source, "to": e.target, "conditional": bool(e.conditional)}
            for e in compiled.edges
        ],
    }
