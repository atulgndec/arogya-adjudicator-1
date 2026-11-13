"""Audited rupee calculator. Mirrors src/lib/adjudicationCalculator.ts.

The model never does arithmetic: this function does, with Decimal, and every deduction
carries the clause that justifies it. Rounding is half-up to whole rupees (as in the original).

Order of rules (same as the original):
  admissibility (policy dates, waiting periods) -> non-payables -> room / ICU caps -> ambulance
  -> medicines (exempt from the proportional cut) -> proportional cut on other lines
  -> cataract and modern-treatment sub-limits -> 5% co-pay -> sum-insured ceiling.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from ..types.adjudication import (AdjudicationLine, AdjudicationWorksheet, ExtractedClaim, SublimitRecord)

D = Decimal
ONE = D(1) # Decimal("1.00") for rounding to whole rupees 
COPAY_RATE = D("0.05") # 5% co-pay on admissible amount, after all other deductions
ROOM_CAP_PCT, ROOM_CAP_MAX = D("0.02"), D(5000) #2% of sum insured or 5000 per day, whichever is lower
ICU_CAP_PCT, ICU_CAP_MAX = D("0.05"), D(10000) #5% of sum insured or 10000 per day, whichever is lower
CATARACT_PCT, CATARACT_MAX = D("0.25"), D(40000) #25% of sum insured or 40000 per hospitalisation, whichever is lower
AMBULANCE_CAP = D(2000) # Ambulance capped at 2,000 per hospitalisation
MODERN_PROCEDURE_PCT = D("0.5") #50% of sum insured for modern procedures

LISTED_24M = ["cataract", "hernia", "hydrocele", "piles", "fistula", "fissure", "gallbladder", "calculi", "stone",
              "cholelithiasis", "tonsil", "adenoid", "mastoid", "tympanoplasty", "hysterectomy", "benign",
              "varicose", "sinus"] # Listed treatments with 24-month waiting period

LISTED_36M = ["joint replacement", "knee replacement", "hip replacement", "osteoarthritis"] # Listed treatments with 36-month waiting period
MODERN = ["robotic", "balloon sinuplasty", "deep brain", "oral chemo", "immunotherapy", "intra vitreal",
          "stereotactic", "bronchial thermoplasty", "vaporisation of prostrate", "stem cell"] # Listed modern procedures

NON_PAYABLE = [
    "glove", "mask", "crepe bandage", "cotton", "diaper", "sanitary pad", "attendant", "admission kit",
    "registration charge", "file charge", "medical records fee", "administrative", "thermometer", "hot pack",
    "cold pack", "ice pack", "spirometer", "walking aid", "arm sling", "bed pan", "linen", "apron", "drape",
    "spirit", "swab", "mouthwash", "lozenges", "urine container", "urine bag", "infusion pump cost", "vasofix",
    "mineral water", "toiletries", "comfort kit", "pulse oximeter", "vip comfort", "personal hygiene",
    "extra bed", "guest bed", "wifi", "telephone", "disinfectant", "fruit basket", "cosmetic",
] # Non-payable consumables (IRDAI Annexure-A List I)


def rupee(x: Decimal) -> Decimal:
    return x.quantize(ONE, rounding=ROUND_HALF_UP) # Round to whole rupees, half-up

# Return True if the description matches any of the non-payable consumables
def is_non_payable(description: str) -> bool:  
    d = description.lower()
    return any(k in d for k in NON_PAYABLE)

# Check if the waiting period is unserved for the claim, based on the diagnosis, 
# procedure, and policy tenure.
def check_waiting_period(c: ExtractedClaim) -> tuple[str, str]:
    """Return (clause, reason) if a waiting period is unserved, else ('', '').

    Each rule is tested independently. (The original used an if / else-if chain on the months
    bracket, so a knee replacement at 14 months escaped the 36-month rule.)
    """
    if c.is_accident: # Accidental claims are exempt from waiting periods
        return "", ""
    text = f"{c.diagnosis} {c.procedure_name}".lower() # Combine diagnosis and procedure for matching
    m = c.continuous_months_active # Calculate the continuous months the policy has been active at the time of admission
    if m < 1:
        return "Clause 6.3 (Excl03)", "First 30 days waiting period not served (tenure under 1 month). Non-accidental claims excluded."
    if m < 24 and any(k in text for k in LISTED_24M): # Check for listed treatments with 24-month waiting period
        return "Clause 6.2(i) (Excl02)", f"Specific 24-month waiting period unserved for '{c.diagnosis}'. Policy active {m} months."
    if m < 36 and any(k in text for k in LISTED_36M): # Check for listed treatments with 36-month waiting period
        return "Clause 6.2(ii) (Excl02)", f"36-month waiting period unserved for joint replacement / osteoarthritis. Policy active {m} months."
    if m < 36 and c.is_pre_existing: # Check for pre-existing conditions with 36-month waiting period
        return "Clause 6.1 (Excl01)", f"36-month waiting period unserved for pre-existing disease ({c.pre_existing_condition_name or 'declared PED'}). Policy active {m} months."
    return "", ""

# Compute a hash of the worksheet, including the claim ID, policy number,
# claimed total, totals, and line items
def worksheet_hash(c: ExtractedClaim, lines: list[AdjudicationLine], totals: tuple) -> str:
    payload = "|".join([c.claim_id, c.policy_number, str(c.claimed_total), *map(str, totals),
                        ";".join(f"{l.line_id}:{l.admissible_amount}" for l in lines)])
    return "SHA256-" + hashlib.sha256(payload.encode()).hexdigest()

# Compute the adjudication worksheet for a claim, applying all rules and caps
def calculate_worksheet(c: ExtractedClaim, *, cataract_previously_used: Decimal = D(0),
                        senior_threshold: Decimal = D(50000)) -> AdjudicationWorksheet: 
    line_sum = sum((i.claimed_amount for i in c.line_items), D(0)) # Sum the claimed amounts of all line items in the claim
    line_sum_ok = abs(line_sum - c.claimed_total) < ONE # Check if the sum of line items matches the claimed total within a rupee
    # Check if the admission date is within the policy period
    in_policy = date.fromisoformat(c.policy_start) <= date.fromisoformat(c.admission_date) <= date.fromisoformat(c.policy_end) 
    wait_clause, wait_reason = check_waiting_period(c)  # Check if any waiting period is unserved for the claim
    inadmissible = (not in_policy) or bool(wait_clause) # Determine if the claim is inadmissible due to policy dates or waiting periods
    
    if not in_policy: # If the admission date is outside the policy period, set the reason and clause for inadmissibility
        why, why_clause = (f"Admission date {c.admission_date} falls outside the policy window {c.policy_start} to {c.policy_end}.",
                           "Clause 2 (Operative Clause)")
    else:
        why, why_clause = wait_reason, wait_clause

    room_cap = min(ROOM_CAP_PCT * c.sum_insured, ROOM_CAP_MAX) # Calculate the room rent cap based on the sum insured and maximum limit
    icu_cap = min(ICU_CAP_PCT * c.sum_insured, ICU_CAP_MAX) # Calculate the ICU rent cap based on the sum insured and maximum limit
    room_breach = c.actual_room_rent_per_day > room_cap # Check if the actual room rent per day exceeds the room rent cap
    icu_breach = c.actual_icu_rent_per_day > icu_cap # Check if the actual ICU rent per day exceeds the ICU rent cap
    ratio = min(ONE, room_cap / c.actual_room_rent_per_day) if room_breach else ONE   # Clause 4.1 Note b # Calculate the proportional cut ratio for room rent if there is a breach
    
    # Initialize a dictionary to track deductions by reason, with all values set to zero
    by_reason = {k: D(0) for k in ["room_rent_cap", "proportional_cut", "icu_cap", "non_payable_consumables",
                                   "waiting_period", "sublimits", "policy_lapse", "copay"]}
    lines: list[AdjudicationLine] = [] # Initialize a list to hold the adjudication lines for each line item in the claim
    sublimits: list[SublimitRecord] = [] # Initialize a list to hold any sublimit records that are applied during adjudication
    text = f"{c.procedure_name} {c.diagnosis}".lower() # Combine the procedure name and diagnosis for matching against sublimits
    is_cataract = "cataract" in text # Check if the claim is for a cataract procedure
    
    # Process each line item in the claim to determine admissibility, deductions, and applicable clauses
    for it in c.line_items:
        claimed = it.claimed_amount
        admissible, reason, clause, prop, note = claimed, "Covered under policy terms", "Clause 4.1", False, ""
        
        # Handle inadmissibility, non-payables, room/ICU caps, ambulance, medicines, and proportional cuts

        if inadmissible:
            admissible, reason, clause = D(0), why, why_clause # If the claim is inadmissible, set admissible amount to zero and use the reason and clause determined earlier
            by_reason["policy_lapse" if not in_policy else "waiting_period"] += claimed # Update the deductions by reason for policy lapse or waiting period based on the inadmissibility reason
            note = f"Repudiated: {why} ({why_clause})." # Add a note indicating that the claim is repudiated due to the reason and clause
        elif it.category == "consumables" or is_non_payable(it.raw_description): # If the line item is a consumable or matches the non-payable list, set admissible amount to zero and update deductions by reason
            admissible, clause = D(0), "Clause 4.7 / Annexure-A List I"
            reason = "Non-payable consumable (IRDAI Annexure-A List I)."
            by_reason["non_payable_consumables"] += claimed # Update the deductions by reason for non-payable consumables
            note = f"100% deduction of {claimed} for non-payable item '{it.raw_description}'."
        elif it.category in ("room", "icu"): # If the line item is a room or ICU charge, apply the respective cap and calculate admissible amount based on days or units
            cap, key, label = (room_cap, "room_rent_cap", "Clause 4.1.a") if it.category == "room" else (icu_cap, "icu_cap", "Clause 4.1.b")
            days = it.days_or_units or D(c.hospitalization_days if it.category == "room" else c.icu_days) or ONE # Use the specified days or units for the line item, or fallback to hospitalization days or ICU days from the claim, defaulting to 1 if not available
            allowed = cap * days # Calculate the allowed amount based on the cap and number of days or units
            clause = label # Set the clause for room or ICU cap based on the category of the line item

            if claimed > allowed: # If the claimed amount exceeds the allowed amount based on the cap and days, set the admissible amount to the allowed amount, update deductions by reason, and add a note indicating the deduction
                admissible = allowed
                reason = f"{'Room' if it.category == 'room' else 'ICU'} cap exceeded (allowed {cap}/day x {days} days = {allowed})."
                by_reason[key] += claimed - allowed
                note = f"Claimed {claimed} - allowed {allowed} = {claimed - allowed} deducted ({label})."
        elif it.category == "ambulance":
            clause = "Clause 4.1.1.e"
            if claimed > AMBULANCE_CAP:
                admissible, reason = AMBULANCE_CAP, "Ambulance capped at 2,000 per hospitalisation."
                by_reason["sublimits"] += claimed - AMBULANCE_CAP # Update the deductions by reason for sublimits for ambulance cap
        elif it.category == "medicines":
            clause = "Clause 4.1.d & Clause 4.1 Note b"       # exempt from the proportional cut
            note = "Medicines are exempt from the room-rent proportional cut (Note b)." if room_breach else ""
        elif room_breach: # If there is a room rent breach, apply a proportional cut to the line item based on the ratio of the room cap to the actual room rent per day, and update deductions by reason for proportional cut
            admissible = rupee(claimed * ratio)
            prop, clause = True, "Clause 4.1 Note b"
            reason = f"Proportional cut: room cap {room_cap} / actual {c.actual_room_rent_per_day} = {ratio * 100:.1f}%."
            note = f"{claimed} x {room_cap}/{c.actual_room_rent_per_day} = {admissible}; deducted {claimed - admissible}."
            by_reason["proportional_cut"] += claimed - admissible
        else:
            clause = "Clause 4.1.c" if it.category in ("surgeon_fee", "doctor_consultation") else "Clause 4.1.d"
        
        # Append the adjudication line for the current line item, including the line ID, raw description, 
        # category, claimed amount, admissible amount, deduction amount, deduction reason, clause cited, 
        # proportional cut applied, proportional ratio (if applicable), and audit explanation (note) 

        lines.append(AdjudicationLine(
            line_id=it.id, raw_description=it.raw_description, category=it.category, claimed_amount=claimed,
            admissible_amount=admissible, deduction_amount=claimed - admissible, deduction_reason=reason if claimed != admissible else "",
            clause_cited=clause, proportional_cut_applied=prop, proportional_ratio=ratio if prop else None, audit_explanation=note))
    
    # To cut amounts from lines based on the specified criteria, 
    # updating the admissible amount, deduction amount, clause cited, and 
    # deduction reason for each line item as needed
    def cut_lines(amount: Decimal, only: tuple | None, clause: str, why_: str) -> None:
        for l in lines:
            if amount <= 0:
                break
            if l.admissible_amount > 0 and (only is None or l.category in only):
                cut = min(l.admissible_amount, amount) # Calculate the cut amount as the minimum of the line's admissible amount and the remaining amount to be cut
                l.admissible_amount -= cut
                l.deduction_amount += cut
                l.clause_cited += f" & {clause}"
                l.deduction_reason = why_
                amount -= cut
    # Apply cataract sub-limit (Clause 4.3) if the claim is for a cataract procedure and is admissible, 
    # calculating the available amount based on the sum insured and previously used amount, 
    # and cutting amounts from relevant line items if the admissible amount exceeds the available limit
    if is_cataract and not inadmissible:                                   # Clause 4.3
        cap = min(CATARACT_PCT * c.sum_insured, CATARACT_MAX)
        available = max(D(0), cap - cataract_previously_used)
        admissible_now = sum((l.admissible_amount for l in lines), D(0))
        if admissible_now > available:
            over = admissible_now - available
            by_reason["sublimits"] += over
            sublimits.append(SublimitRecord(sublimit_key="cataract", name="Cataract sub-limit (Clause 4.3)", clause="Clause 4.3", max_cap=cap,
                                            claimed=admissible_now, previously_utilized=cataract_previously_used, deduction=over, admissible=available))
            cut_lines(over, ("procedure", "surgeon_fee", "operation_theatre"), "Clause 4.3",
                      f"Cataract sub-limit {cap} (already used {cataract_previously_used}) leaves {available}.")
    # Apply modern procedure sub-limit (Clause 4.6) if the claim involves any modern procedures and is admissible,
    # calculating the cap based on the sum insured, and cutting amounts from relevant line items
    # if the admissible amount exceeds the cap
    if any(k in text for k in MODERN) and not inadmissible:                # Clause 4.6
        cap = MODERN_PROCEDURE_PCT * c.sum_insured
        now = sum((l.admissible_amount for l in lines), D(0))
        if now > cap:
            over = now - cap
            by_reason["sublimits"] += over
            sublimits.append(SublimitRecord(sublimit_key="modern_procedure", name="Modern procedure cap, 50% of SI (Clause 4.6)", clause="Clause 4.6",
                                            max_cap=cap, claimed=now, previously_utilized=D(0), deduction=over, admissible=cap))
            cut_lines(over, None, "Clause 4.6", "Capped at 50% of sum insured for modern procedures.")
    
    # Compute the total deductions, admissible amount before co-pay, co-pay amount, 
    # and final payable amount based on the claim's coverage limit and sum insured    
    
    total_ded = sum((l.deduction_amount for l in lines), D(0))
    admissible_before = max(D(0), c.claimed_total - total_ded)
    copay = D(0) if inadmissible else rupee(admissible_before * COPAY_RATE)
    by_reason["copay"] = copay
    ceiling = c.total_coverage_limit or (c.sum_insured + c.cumulative_bonus)
    final = min(max(D(0), admissible_before - copay), ceiling)
    
    # Determine if senior approval is required based on the final payable amount, 
    # claimed total, and any breaches of room rent cap or coverage ceiling
    
    senior: list[str] = []
    if final > senior_threshold or c.claimed_total > senior_threshold:
        senior.append(f"Claim value exceeds the {senior_threshold} high-value threshold.")
    if room_breach:
        senior.append(f"Room rent exceeds the cap by {c.actual_room_rent_per_day - room_cap}/day (proportional deduction applied).")
    if c.claimed_total > ceiling:
        senior.append(f"Claimed amount {c.claimed_total} exceeds the coverage ceiling {ceiling} (Clause 2).")
    if not line_sum_ok:
        senior.append(f"Bill total {c.claimed_total} differs from the sum of its lines {line_sum}.")

    return AdjudicationWorksheet(
        claim_id=c.claim_id, worksheet_id=f"WS-{c.claim_id}", policy_number=c.policy_number, insured_name=c.insured_name,
        claimed_total=c.claimed_total, line_sum_total=line_sum, line_sum_matches_bill=line_sum_ok, is_admissible=not inadmissible,
        inadmissibility_reason=why if inadmissible else None, inadmissibility_clause=why_clause if inadmissible else None,
        room_rent_cap_per_day=room_cap, actual_room_rent_per_day=c.actual_room_rent_per_day, room_rent_breach=room_breach,
        proportional_cut_ratio=ratio, icu_rent_cap_per_day=icu_cap, actual_icu_rent_per_day=c.actual_icu_rent_per_day, icu_rent_breach=icu_breach,
        lines=lines, total_deductions=total_ded, admissible_before_copay=admissible_before, copay_amount=copay, final_payable_amount=final,
        sublimits_applied=sublimits, deductions_by_reason=by_reason,
        worksheet_hash=worksheet_hash(c, lines, (total_ded, admissible_before, copay, final)),
        requires_senior_approval=bool(senior), senior_approval_reasons=senior)

# Recompute totals after an adjudicator edits lines, 
# invalidating earlier approvals and updating the worksheet hash
# 
# This function is designed to be called after a human adjudicator 
# has manually edited the AdjudicationLine objects
def recompute_totals(c: ExtractedClaim, ws: AdjudicationWorksheet, lines: list[AdjudicationLine]) -> AdjudicationWorksheet:
    """After an adjudicator edits lines: recompute totals and the hash (which invalidates earlier approvals)."""
    total_ded = sum((l.deduction_amount for l in lines), D(0))
    admissible_before = max(D(0), c.claimed_total - total_ded)
    copay = rupee(admissible_before * COPAY_RATE) if ws.is_admissible else D(0)
    final = min(max(D(0), admissible_before - copay), c.total_coverage_limit)
    return ws.model_copy(update=dict(
        lines=lines, total_deductions=total_ded, admissible_before_copay=admissible_before, copay_amount=copay, final_payable_amount=final,
        worksheet_hash=worksheet_hash(c, lines, (total_ded, admissible_before, copay, final)) + f"-REV{ws.revisions_count + 1}",
        revisions_count=ws.revisions_count + 1, approval_state="invalidated", approval_token=None, settlement_issued=False,
        critic_status="not_rerun", critic_notes="Edited by an adjudicator; the critic was not re-run. Totals were recomputed; earlier approval is void."))
