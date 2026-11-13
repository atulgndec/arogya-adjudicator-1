"""The claim workflow as a real LangGraph graph. Replaces src/lib/langgraph.ts (a hand-made StateGraph).

supervisor -> extract -> fraud_screen -> coverage -> adjudicator -> critic --(fail, once)--> coverage
                                                                      |
                                              human (interrupt) <-----+---> edit loops back to human
                                                   | approve                | reject -> memory
                                                 settle -> memory
Money is computed only by adjudication_calculator. The model never changes a number.
"""
from __future__ import annotations

import operator
from decimal import Decimal
from typing import Annotated, Any, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..types.adjudication import AdjudicationWorksheet, ExtractedClaim, FraudReport
from . import llm
from .adjudication_calculator import calculate_worksheet, recompute_totals
from .claims_db import claims_db, now
from .claims_metrics import CLAIMS, DEDUCTED, DECISIONS, CRITIC_REVISIONS, FRAUD_FLAGS, SETTLED, timed
from .fraud_detector import detect_fraud
from .vector_db import hybrid_clauses
from .privacy import mask, set_phi
from .reflexion_engine import reflexion_engine

D = Decimal
COMPLEX_LIMIT = D(100000) 
# Claims above this amount are considered complex and require model adjudication.


class State(TypedDict, total=False):
    claim: ExtractedClaim
    fraud: FraudReport
    ws: AdjudicationWorksheet # The worksheet is updated in-place, but we keep a copy of the previous version to track revisions.
    auto_approve: Optional[dict]          # {"approver": ..., "level": "junior"|"senior"} for batch runs and tests
    lessons: list[str] # IDs of lessons retrieved from the models
    outcome: str                          # settled | repudiated | rejected | pending
    settlement: dict 
    steps: Annotated[list[dict], operator.add]   # audit trail; masked before it is stored


def step(node: str, summary: str, **extra) -> dict:
    return {"steps": [{"node": node, "at": now(), "summary": mask(summary), **extra}]}


# ---- nodes -------------------------------------------------------------------------------
def supervisor(s: State):
    c = s["claim"] # should already have the claim extracted from the PDF and stored in the state. We just log it and set PHI for the audit trail.
    set_phi(c) # mark the claim as containing PHI so it is masked in the audit trail
    return step("supervisor", f"Routing claim {c.claim_id}: {len(c.line_items)} bill lines, claimed ₹{c.claimed_total}.")


def extract(s: State):
    c = s["claim"] # should already have the claim extracted from the PDF and stored in the state. We just log it and set PHI for the audit trail.
    got = sum((i.claimed_amount for i in c.line_items), D(0)) # sum the claimed amounts of all line items in the claim
    ok = got == c.claimed_total
    return step("extract", f"Line items sum to ₹{got}; claim form says ₹{c.claimed_total}. {'Match.' if ok else 'MISMATCH, flagged for the adjudicator.'}")


def fraud_screen(s: State): # It detects fraud in the claim using the detect_fraud function, which returns a FraudReport object containing flags and a risk score. The function then increments the corresponding metrics for each fraud flag detected and returns the fraud report along with a summary of the risk score and number of flags found.
    with timed("fraud_screen"): # timed? It is a context manager that measures the time taken to execute the code block within it. In this case, it measures the time taken to perform the fraud screening process.
        r = detect_fraud(s["claim"])
    for f in r.flags: # It iterates over the flags in the fraud report and increments the corresponding metric for each flag type detected. 
                      #The FRAUD_FLAGS.labels(f.type).inc() line increments the counter for the specific fraud flag type.
        FRAUD_FLAGS.labels(f.type).inc()
    return {"fraud": r, **step("fraud_screen", f"Risk {r.risk_score}/100, {len(r.flags)} flag(s): {', '.join(f.type for f in r.flags) or 'none'}.")}

#The function needs both the claim and the fraud report to calculate the coverage worksheet. 
# The claim contains the details of the claim, while the fraud report contains information 
# about any detected fraud flags and risk score. 
# Both pieces of information are necessary to determine the coverage and deductions for the claim.
def coverage(s: State):
    c, r = s["claim"], s["fraud"] 
    with timed("coverage"): # Coverage is computed in one pass, and the worksheet is immutable once created. Revisions are tracked by incrementing a counter.
        # It calculates the coverage worksheet for the claim, taking into account any previously used sublimits for cataract procedures. 
        # The calculate_worksheet function computes the deductions, co-pay, and final payable amount based
        # on the claim details and any applicable rules or policies.
        ws = calculate_worksheet(c, cataract_previously_used=claims_db.used_sublimits(c.policy_number).get("cataract", D(0))) 
    prev = s.get("ws") # If there is a previous worksheet in the state, it increments the revisions count by 1. This allows tracking of how many times the worksheet has been revised during the adjudication process.
    if prev:
        ws = ws.model_copy(update={"revisions_count": prev.revisions_count + 1}) # It creates a new worksheet with the revisions_count incremented by 1.
    if r.has_fraud_flag:
        ws.requires_senior_approval = True
        ws.senior_approval_reasons = [*ws.senior_approval_reasons, f"Fraud screen: {len(r.flags)} flag(s), risk {r.risk_score}."]
    return {"ws": ws, **step("coverage", f"Deductions ₹{ws.total_deductions}, co-pay ₹{ws.copay_amount}, payable ₹{ws.final_payable_amount}.")}

# This function determines whether a claim is complex or straightforward based on various triggers,
# such as high-value claims, room rent breaches, pre-existing conditions, fraud flags, 
# and mismatched bill totals. If the claim is straightforward, it skips the model call and 
# returns the worksheet and lessons. If the claim is complex, it retrieves relevant clauses 
# using BM25 and vector databases, prepares structured data for the model, and calls the LLM 
# to analyze the case. The results are stored in the worksheet along with any lessons learned.
def adjudicator(s: State):
    """Complex cases get a BM25-grounded model note. Simple ones skip the model entirely."""
    c, ws = s["claim"], s["ws"] # The function retrieves the claim and worksheet from the state dictionary. The claim contains information about the patient's medical condition, procedures, and claimed amounts, while the worksheet contains calculated deductions, co-pay amounts, and final payable amounts based on the claim details.
    triggers = [] 
    
    # store any conditions that indicate the claim is complex. 
    #These triggers will be used to determine whether the claim requires further analysis
    # by the model or can be processed as a straightforward case.
    
    if ws.final_payable_amount >= COMPLEX_LIMIT: triggers.append("High-value claim")
    if ws.room_rent_breach: triggers.append("Room rent breach with proportional cut")
    if c.is_pre_existing: triggers.append("Pre-existing condition declared")
    if s["fraud"].has_fraud_flag: triggers.append("Fraud or attack flag")
    if not ws.line_sum_matches_bill: triggers.append("Bill total does not match line items")
    lessons = reflexion_engine.retrieve(c, ws)
    # It retrieves any lessons learned from previous cases that are relevant to the current
    # claim and worksheet. These lessons can provide insights or guidance for the adjudicator
    # when analyzing the case.
    ws.is_complex_case, ws.complex_case_triggers = bool(triggers), triggers
    if not triggers: # If there are no triggers indicating complexity, 
        # the function increments the "skipped" counter for LLM calls and returns the worksheet, 
        # lessons, and a step indicating that it is a straightforward case with no model call.
        llm.LLM_CALLS.labels("skipped").inc()
        return {"ws": ws, "lessons": [l.id for l in lessons], **step("adjudicator", "Straightforward case: rules only, no model call.", llm=False)}
    
    # If there are triggers indicating complexity, the function increments the "called" 
    # counter for LLM calls and proceeds to retrieve relevant clauses using BM25 
    # and vector databases. It prepares structured data for the model, including claim details, 
    # deductions, and flags. The function also constructs a fallback message summarizing the 
    # triggers and retrieved clauses. Finally, it calls the LLM to analyze the case,
    # storing the results in the worksheet along with any lessons learned. The function 
    # returns the updated worksheet, lessons, and a step indicating that it is a complex 
    # case with a model call.
    
    docs = hybrid_clauses(" ".join([c.diagnosis, c.procedure_name, *triggers]), k=4)      # BM25 + vector database 
    # The function retrieves relevant clauses from the hybrid clause database based on the
    # claim's diagnosis, procedure name, and identified triggers. It uses a combination of 
    # BM25 and vector database techniques to find the most relevant clauses that may apply 
    # to the case. The retrieved clauses are stored in the `docs` variable.
    clauses = "\n".join(f"{d.metadata['clause']} {d.metadata['title']}: {d.page_content}" for d in docs)
    # The function constructs a string containing the retrieved clauses, including their 
    # clause numbers, titles, and page content. This string is stored in the `clauses` variable 
    # and will be used as input for the LLM to analyze the case.
    # Only structured fields go to the model. Raw document text (where an attack could hide) 
    # does not.
    data = {"claimId": c.claim_id, "age": c.patient_age, "diagnosis": c.diagnosis, "procedure": c.procedure_name,
            "claimed": str(c.claimed_total), "payable": str(ws.final_payable_amount), "deductions": {k: str(v) for k, v in ws.deductions_by_reason.items() if v},
            "flags": [f.title for f in s["fraud"].flags]} 
    # The function prepares structured data to be sent to the LLM for analysis. 
    # It includes the claim ID, patient age, diagnosis, procedure name, claimed amount, 
    # final payable amount, deductions by reason, and any fraud flags detected.
    # This structured data is stored in the `data` variable and will be used as input for the 
    # LLM to analyze the case.
    fallback = "Rules-only analysis. Triggers: " + "; ".join(triggers) + ". Clauses retrieved: " + ", ".join(d.metadata["clause"] for d in docs) + "." 
    # fallback message summarizing the triggers and retrieved clauses. 
    # This message will be used as a fallback in case the LLM analysis fails or is inconclusive.
    out = llm.analyze_case(triggers=triggers, clauses=clauses, lessons="\n".join(f"{l.id}: {l.lesson_rule}" for l in lessons), claim_data=data, fallback=fallback) 
    # out is a dict with keys: model, is_ai, summary, lessons
    # The function calls the LLM to analyze the case, passing in the identified triggers,
    # retrieved clauses, lessons learned, structured claim data, and the fallback message.
    # The LLM will analyze the case and provide a summary, any additional lessons learned,
    # and information about the model used for analysis. The results are stored in the `out` variable.      
    
    ws.llm_adjudication = out 
    # The function stores the LLM's adjudication results in the worksheet's `llm_adjudication` field.
    return {"ws": ws, "lessons": [l.id for l in lessons], **step("adjudicator", f"Complex case ({'; '.join(triggers)}). Model: {out['model']}.", llm=out["is_ai"])} 
    # The function returns the updated worksheet, lessons learned, and a step indicating
    # that it is a complex case with a model call. The step includes information about the 
    # identified triggers and the model used for analysis.

# The critic function performs independent checks on the worksheet to ensure that all 
# deductions have corresponding clauses cited, that medicines are not cut proportionally
# in violation of policy, and that the total deductions and admissible amounts match the 
# claimed total. It also checks that the final payable amount is correctly calculated 
# based on the admissible amount and co-pay. If any discrepancies are found, the worksheet's 
# critic status is set to "revision_requested," and a list of problems is recorded. 
# If no problems are found, the status is set to "passed." The function returns the 
# updated worksheet along with a summary of the checks performed.
def critic(s: State):
    """Independent checks on the worksheet. One send-back, then a human decides."""
    ws, c = s["ws"], s["claim"] 
    # The function retrieves the worksheet and claim from the state dictionary. 
    # The worksheet contains the calculated deductions, co-pay amounts, and final payable 
    # amounts based on the claim details, while the claim contains information about 
    # the patient's medical condition, procedures, and claimed amounts.
    problems = [] 
    # The function initializes an empty list called `problems` 
    #to store any discrepancies or issues found during the checks performed on the worksheet. 
    # If any problems are identified, they will be added to this list for further review and action.


    if any(l.deduction_amount > 0 and not l.clause_cited for l in ws.lines): 
        #  does any line item have a deduction amount greater than 0 and no clause cited? 
        # If so, add a problem to the list.
        problems.append("A deduction has no clause cited.")
    if ws.room_rent_breach and any(l.category == "medicines" and l.deduction_amount > 0 and l.deduction_reason and "proportional" in l.deduction_reason.lower() for l in ws.lines): 
        # If there is a room rent breach and any line item in the worksheet is 
        # categorized as "medicines" with a deduction amount greater than 0, a deduction reason 
        # provided, and the word "proportional" in the deduction reason (case-insensitive), 
        # then add a problem to the list.
        problems.append("Medicines were cut proportionally (exempt under Clause 4.1 Note b).")
    if ws.total_deductions + ws.admissible_before_copay != c.claimed_total: 
        # If the sum of total deductions and admissible amount before co-pay does not equal the
        #  claimed total, add a problem to the list.
        problems.append("Deductions plus admissible amount do not equal the claimed total.")
    if ws.admissible_before_copay - ws.copay_amount != ws.final_payable_amount and ws.final_payable_amount != c.total_coverage_limit:
        # If the admissible amount before co-pay minus the co-pay amount does not equal the final 
        # payable amount, and the final payable amount is not equal to the total coverage limit, 
        # add a problem to the list.
        problems.append("Payable amount does not equal admissible minus co-pay.")
    ws.critic_status = "revision_requested" if problems else "passed"
    ws.critic_discrepancies, ws.critic_notes = problems, "; ".join(problems) or "All checks passed."
    if problems and ws.revisions_count < 1: 
        # If there are problems and the revisions count is less than 1, increment the 
        # CRITIC_REVISIONS metric.
        CRITIC_REVISIONS.inc() 
        # The function increments the CRITIC_REVISIONS metric to track the number of times 
        # the critic has requested revisions for a worksheet. This metric can be used for 
        # monitoring and analysis of the adjudication process.
    return {"ws": ws, **step("critic", ws.critic_notes)} 

# returns the next node in the graph based on the critic's status and the number of revisions made.
def after_critic(s: State) -> str:
    ws = s["ws"]
    return "coverage" if ws.critic_status == "revision_requested" and ws.revisions_count < 1 else "human" 
# if the critic requested a revision and the revisions count is less than 1, 
# go back to coverage; otherwise, go to human for adjudication.


# The human function handles the adjudication process by a human adjudicator. 
# It checks if the worksheet is admissible, and if not, it returns a repudiated outcome. 
# If the worksheet is admissible, it interrupts the process to allow for human review 
# and decision-making. The decision can be to approve, reject, or edit the claim. 
# The function updates the worksheet and logs the decision accordingly. It also handles 
# senior approval requirements and ensures that the four-eyes rule is followed. 
# Finally, it returns the updated worksheet and outcome based on the human 
# adjudicator's decision.

def human(s: State):
    """Pause for an adjudicator. interrupt() stores the state; Command(resume=...) continues it."""
    ws, c = s["ws"], s["claim"]
    if not ws.is_admissible: # claim has been found to be invalid or not meeting the necessary criteria for approval. In this case, the function returns a repudiated outcome, indicating that the claim has been rejected automatically without further human review.
        return {"outcome": "repudiated", **step("human", f"Repudiation reviewed automatically: {ws.inadmissibility_reason}")}
    decision = s.get("auto_approve") or interrupt({"claim_id": c.claim_id, "worksheet_hash": ws.worksheet_hash, "payable": str(ws.final_payable_amount),
                                                   "requires_senior": ws.requires_senior_approval, "reasons": ws.senior_approval_reasons})  
    # wait for a human to approve, reject, or edit the claim. The interrupt function stores the 
    # current state and allows for human intervention. The decision can be provided through 
    # auto_approve (for batch runs) or through the interrupt mechanism.
    action, who, level = decision.get("action", "approve"), decision["approver"], decision.get("level", "junior")
    DECISIONS.labels(action, level).inc()
    if action == "reject":
        claims_db.log_decision(c.claim_id, "reject", who, level)
        ws.approval_state, ws.rejection_reason = "rejected", decision.get("reason", "")
        return {"ws": ws, "outcome": "rejected", **step("human", f"Rejected by {who}: {ws.rejection_reason}")}
    if action == "edit":
        lines, n = [l.model_copy() for l in ws.lines], 0
        for e in decision["edits"]:
            l = next(x for x in lines if x.line_id == e["line_id"])
            original = l.deduction_amount
            l.deduction_amount, l.admissible_amount = D(str(e["deduction_amount"])), l.claimed_amount - D(str(e["deduction_amount"]))
            l.edited_by_adjudicator, l.adjudicator_comment = True, e.get("comment", "")
            reflexion_engine.propose_from_edit(c, l, original, l.adjudicator_comment)
            n += 1
        reflexion_engine.record_outcome(s.get("lessons", []), edited=True)
        claims_db.log_decision(c.claim_id, "edit", who, level, n)
        # Auto-approval is a one-shot for batch runs; after an edit a person must approve the new version.
        return {"ws": recompute_totals(c, ws, lines), "auto_approve": None, **step("human", f"{who} edited {n} line(s); earlier approval voided, new hash issued.")}
    if ws.requires_senior_approval and level != "senior":
        return {**step("human", f"{who} ({level}) cannot approve: senior approval is required ({'; '.join(ws.senior_approval_reasons)}).")}
    if ws.requires_senior_approval and ws.junior_adjudicator_name and who == ws.junior_adjudicator_name:
        return {**step("human", f"Four-eyes rule: {who} already reviewed this claim and cannot also give the senior approval.")}
    ws.approval_state = "senior_approved" if level == "senior" else "junior_approved"
    ws.junior_adjudicator_name = ws.junior_adjudicator_name or who
    if level == "senior": ws.senior_adjudicator_name = who
    ws.approval_token = claims_db.issue_token(c.claim_id, ws.worksheet_hash, who, level)
    claims_db.log_decision(c.claim_id, "approve", who, level)
    reflexion_engine.record_outcome(s.get("lessons", []), edited=False)
    return {"ws": ws, "outcome": "approved", **step("human", f"Approved by {who} ({level}); token bound to worksheet {ws.worksheet_hash[:12]}.")}

# after_human function determines the next step in the claims processing workflow based on the
#  outcome of the human adjudication. If the outcome is "approved," it proceeds to the "settle"
#  step. If the outcome is either "rejected" or "repudiated," it moves to the "memory" step for 
# audit and record-keeping. If the outcome is neither of these, it returns to the "human" step 
# for further review or edits, indicating that additional human intervention is required.
def after_human(s: State) -> str:
    ws, out = s["ws"], s.get("outcome", "")
    if out == "approved": return "settle"
    if out in ("rejected", "repudiated"): return "memory"
    return "human"          # edited, or approval refused: ask again


def settle(s: State):
    ws, c = s["ws"], s["claim"]
    sub = {r.sublimit_key: float(r.admissible) for r in ws.sublimits_applied}
    res = claims_db.settle_claim(caller="Settlement", claim_id=c.claim_id, policy_no=c.policy_number, claimant=c.insured_name, expected_name=ws.insured_name,
                                 token=ws.approval_token, ws_hash=ws.worksheet_hash, amount=ws.final_payable_amount, bill_no=c.bill_number, sublimits_used=sub)
    if not res["success"]:
        return {"settlement": res, "outcome": "blocked", **step("settle", res["error"])}
    ws.settlement_issued = True
    SETTLED.inc(float(ws.final_payable_amount))
    return {"ws": ws, "settlement": res, "outcome": "settled", **step("settle", f"Settled ₹{ws.final_payable_amount} ({res['settlementId']}).")}


def memory(s: State):
    ws = s["ws"]
    out = s.get("outcome", "pending")
    CLAIMS.labels(out).inc()
    for k, v in ws.deductions_by_reason.items():
        if v: DEDUCTED.labels(k).inc(float(v))
    return step("memory", f"Outcome {out} recorded for audit and the playbook.")

# Build the claims processing graph
def build_graph(checkpointer=None):
    g = StateGraph(State)
    for name, fn in [
                    ("supervisor", supervisor), 
                    ("extract", extract), 
                    ("fraud_screen", fraud_screen), 
                    ("coverage", coverage),
                    ("adjudicator", adjudicator), 
                    ("critic", critic), 
                    ("human", human),
                    ("settle", settle), 
                    ("memory", memory)
                    ]:
        g.add_node(name, fn)

    g.add_edge(START, "supervisor")
    for a, b in [
                ("supervisor", "extract"), 
                ("extract", "fraud_screen"), 
                ("fraud_screen", "coverage"), 
                ("coverage", "adjudicator"), 
                ("adjudicator", "critic"), 
                ("settle", "memory"), 
                ("memory", END)]:
        g.add_edge(a, b)
    g.add_conditional_edges(
        "critic", after_critic, {"coverage": "coverage", "human": "human"})
    g.add_conditional_edges(
        "human", after_human, {"settle": "settle", "memory": "memory", "human": "human"})
    # Allow our own pydantic models in checkpoints (silences the unregistered-type warning, stays strict for others).
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("arogya.types.adjudication", n) for n in ("ExtractedClaim", "FraudReport", "AdjudicationWorksheet")]) 
    # allow our own pydantic models in checkpoints (silences the unregistered-type warning, stays strict for others).
    return g.compile(checkpointer=checkpointer or MemorySaver(serde=serde))
