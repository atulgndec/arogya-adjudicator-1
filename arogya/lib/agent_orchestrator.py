"""Thin layer over the graph. Mirrors src/lib/agentOrchestrator.ts: run a claim, resume it, write the letter."""
from __future__ import annotations

from langgraph.types import Command

from ..data.synthetic_claims import get_claim
from . import llm
from .claims_graph import build_graph
from .claims_metrics import PENDING

graph = build_graph()

# config for the graph, used as a key to store state
def _cfg(claim_id: str) -> dict: 
    return {"configurable": {"thread_id": claim_id}}

# snapshot of the current state of a claim, including next steps and outcome
def snapshot(claim_id: str) -> dict:
    st = graph.get_state(_cfg(claim_id))
    v = st.values # v is None if the claim has not been run yet
    if not v:
        return {"status": "not_started"}
    return {"status": "waiting_for_adjudicator" if st.next else "finished", "next": list(st.next), "outcome": v.get("outcome", "pending"),
            "worksheet": v["ws"].model_dump(mode="json", by_alias=True) if v.get("ws") else None,
            "fraud": v["fraud"].model_dump(mode="json", by_alias=True) if v.get("fraud") else None, "steps": v.get("steps", []), "settlement": v.get("settlement")}

# Refresh the PENDING metric to reflect the number of claims 
# that are currently waiting for adjudication
def _refresh_pending() -> None:
    PENDING.set(sum(1 for t in _threads if graph.get_state(_cfg(t)).next))

# Keep track of the claim IDs that are currently being processed
_threads: set[str] = set()

# execute the claim through the graph, which will run the adjudication process 
# and produce an outcome
# returns a snapshot of the claim's current state after processing
# auto_approve is an optional dictionary that can be used to automatically 
# approve certain steps in the adjudication process
def run_claim(claim_id: str, auto_approve: dict | None = None) -> dict:
    claim = get_claim(claim_id) # Retrieve the claim data from the synthetic_claims.py
    _threads.add(claim_id) # Add the claim ID to the set of threads that are currently being processed
    graph.invoke({"claim": claim, "auto_approve": auto_approve}, _cfg(claim_id)) # invoke the graph with the claim data and optional auto_approve parameter
    _refresh_pending()
    return snapshot(claim_id)

# decide on a claim that is waiting for adjudication. 
# The decision can be to approve, edit, or reject the claim, and may include
# additional information such as the approver, level, edits, and reason.
def decide(claim_id: str, decision: dict) -> dict:
    """decision: {action: approve|edit|reject, approver, level, edits?, reason?}"""
    if not graph.get_state(_cfg(claim_id)).next:
        raise ValueError("claim is not waiting for a decision")
    graph.invoke(Command(resume=decision), _cfg(claim_id))
    _refresh_pending()
    return snapshot(claim_id)

# Generate a settlement letter for a claim that has been processed.
# The letter includes details about the claim, deductions, co-pay, and final payable amount.
def settlement_letter(claim_id: str) -> dict:
    v = graph.get_state(_cfg(claim_id)).values
    if not v or not v.get("ws"): # Check if the claim has been processed and has a worksheet; if not, raise an error
        raise ValueError("run the claim first")
    ws, c = v["ws"], v["claim"]
    fallback = (f"# Settlement advice: {c.claim_id}\n\nClaimed ₹{c.claimed_total}; deductions ₹{ws.total_deductions}; co-pay ₹{ws.copay_amount}; "
                f"**payable ₹{ws.final_payable_amount}**.\n\n" + "\n".join(f"- {l.raw_description}: ₹{l.deduction_amount} ({l.clause_cited})" for l in ws.lines if l.deduction_amount > 0)
                + "\n\nIf you disagree, write to the insurer's grievance officer, then the Insurance Ombudsman.")
    return llm.settlement_report(claim_data={"claimId": c.claim_id, "diagnosis": c.diagnosis, "procedure": c.procedure_name},
                                 worksheet={"payable": str(ws.final_payable_amount), "deductions": [(l.raw_description, str(l.deduction_amount), l.clause_cited) for l in ws.lines if l.deduction_amount > 0]}, fallback=fallback)
