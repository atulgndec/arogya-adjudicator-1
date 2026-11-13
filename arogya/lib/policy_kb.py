"""Policy knowledge base: numbered clauses, bill-term normalisation and BM25 retrieval.

Mirrors src/lib/policyKb.ts and src/lib/vectorDb.ts. The original used a hand-made 64-dimension
"embedding"; this uses BM25 (rank_bm25) behind a LangChain retriever, so the same index can sit in any chain.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi

from ..types.adjudication import PolicyClause

DATA = Path(__file__).parent.parent / "data"
CLAUSES = [PolicyClause.model_validate(c) for c in json.loads((DATA / "policy_clauses.json").read_text())] 
# policy clauses are static, so load them once for all requests
TERM_MAP: dict[str, dict] = json.loads((DATA / "term_map.json").read_text()) 
# hospital bill-term normalisation is static, so load it once for all requests

def tokenize(text: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z0-9.]+", " ", text.lower()).split() if len(t) > 1] 
# tokenize by words, ignore single-character tokens

# clauses are numbered, so we can use the number to retrieve the clause text. The text is used for BM25 retrieval, 
# and the number is used to identify the clause in the response.
def clause_text(c: PolicyClause) -> str:
    # title and tags are repeated so they weigh more than body text (a simple field boost)
    return f"{c.clause_number} {c.title} {c.title} {' '.join(c.tags)} {' '.join(c.tags)} {c.text}" 
# return the clause text, with title and tags repeated for weighting


class ClauseRetriever(BaseRetriever):
    """LangChain retriever over the numbered clauses. Returns Documents carrying the clause number."""
    k: int = 4
    min_score: float = 0.5
    # k is the number of clauses to return, min_score is the minimum BM25 score to return a clause. 
    # Both can be tuned for the use case.
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None) -> list[Document]: 
        """Get the relevant documents for the query, using BM25 retrieval."""   
        index = _index()
        scores = index.get_scores(tokenize(query)) # get the BM25 scores for the query against the indexed clauses
        ranked = sorted(zip(CLAUSES, scores), key=lambda p: p[1], reverse=True)[: self.k] # rank the clauses by score and take the top k
        return [Document(page_content=c.text, metadata={"clause": c.clause_number, "title": c.title, "rule_type": c.rule_type, "score": round(float(s), 2)})
                for c, s in ranked if s >= self.min_score]

# BM25 index is built once and cached, so it can be reused for all queries.
# The index is built over the clause text, which includes the title and tags for weighting.
@lru_cache
def _index() -> BM25Okapi:
    return BM25Okapi([tokenize(clause_text(c)) for c in CLAUSES])


def search_clauses(query: str, k: int = 4) -> list[Document]:
    return ClauseRetriever(k=k).invoke(query)

# get_clause returns the PolicyClause object for a given clause number or id. 
# It normalizes the input to ignore case and punctuation, so "Clause 4.1.d" and "clause 4.1.d" both match the same clause.
def get_clause(number: str) -> PolicyClause | None:
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    return next((c for c in CLAUSES if norm(number) == norm(c.clause_number) or norm(number) in norm(c.id)), None)


def normalize_bill_term(term: str) -> dict:
    """Map a hospital's wording to a standard head, category and clause (the original's Requirement 1)."""
    t = term.lower().strip()
    for key, m in TERM_MAP.items():
        if key in t or t in key:
            return m
    return {"normalized": term, "category": "other", "primaryClause": "Clause 4.1.d"}
