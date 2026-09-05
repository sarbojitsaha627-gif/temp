# app/orchestrator — Deterministic batch pipeline
#
# This package implements the plain-Python batch orchestrator tier
# described in plan.md Decision #2 (Option B).  All deterministic
# processing (ingestion, exact joins, fuzzy scoring, clustering,
# persistence, safety checks, and reporting) lives here.
#
# The LangGraph cluster subgraph (app.agent.graph.run_cluster) is
# invoked only for ambiguous clusters that survive the fast path.
