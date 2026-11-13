"""Masking of health details in traces (the original's Requirement 8), with the regex-escaping bug fixed."""
import re
from contextvars import ContextVar 
#contextvars? 
#It is a module that provides support for managing context-local state, 
#allowing you to store and retrieve data that is specific to the current
#context of execution. This is particularly useful in asynchronous programming,
#where different tasks may need to maintain their own state without
#interfering with each other.

from ..types.adjudication import ExtractedClaim

_PHI: ContextVar[list[tuple[str, str]]] = ContextVar("phi", default=[])


def set_phi(c: ExtractedClaim) -> None:
    pairs = [(c.insured_name, "[PATIENT]"), (c.policy_number, "[POLICY_NO]"), (c.diagnosis, "[DIAGNOSIS]"), (c.icd10_code, "[ICD]"), (c.procedure_name, "[PROCEDURE]")] 
    # masking the patient name, policy number, diagnosis, ICD-10 code, and procedure name with corresponding tags
    _PHI.set([(v, tag) for v, tag in pairs if v])
# Protected Health Information. It refers to any information in a medical context 
# that can be used to identify an individual and relates to their health status, 
# provision of healthcare, or payment for healthcare services. In this code, 
# PHI is being masked to protect patient privacy in audit trails and logs.

def mask(text: str) -> str:
    for value, tag in _PHI.get():
        text = re.sub(re.escape(value), tag, text, flags=re.I) # Escape the value to treat it as a literal string in the regex
    return text
