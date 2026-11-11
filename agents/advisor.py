"""Advisor Agent: grounded savings advice = exact FACTS (code) + retrieved GUIDANCE (ChromaDB) + LLM writing.

Prompting: chain-of-thought in a `reasoning` field, then a REFLECTION pass.
Reflection is triggered only when an automatic check finds a number in the draft that is not in
the facts/guidance, so most answers cost a single LLM call.
"""
from __future__ import annotations
import re
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llm_utils import RobustPydanticParser, with_repair

from prompts import ADVISOR_SYSTEM, ADVISOR_USER, REFLECT_SYSTEM, REFLECT_USER
from agents.analysis import snapshot


class AdvisorOutput(BaseModel):
    reasoning: str = Field(description="private step-by-step notes, max 120 words")
    answer: str = Field(description="markdown answer for the user")


class ReflectOutput(BaseModel):
    unsupported_claims: list[str] = Field(default_factory=list)
    revised_answer: str


_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[float]:
    out = set()
    for m in _NUM.findall(text):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def unsupported_numbers(answer: str, *sources: str) -> list[float]:
    """Numbers >= 100 in the answer with no match (within 1% or 1 rupee) in the sources. Years are ignored."""
    allowed = set().union(*(_numbers(s) for s in sources))
    bad = []
    for n in sorted(_numbers(answer)):
        if n < 100 or (1900 <= n <= 2100 and n == int(n)):
            continue
        if not any(abs(n - a) <= max(1.0, 0.01 * a) for a in allowed):
            bad.append(n)
    return bad


def _chain(llm, system, user, model, nudge_key="question"):
    parser = RobustPydanticParser(pydantic_object=model)
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)]).partial(
        format_instructions=parser.get_format_instructions())
    return with_repair(prompt | llm | parser, nudge_key)


class Advisor:
    def __init__(self, llm, df, rag=None):
        self.df, self.rag = df, rag
        self.advise_chain = _chain(llm, ADVISOR_SYSTEM, ADVISOR_USER, AdvisorOutput)
        self.reflect_chain = _chain(llm, REFLECT_SYSTEM, REFLECT_USER, ReflectOutput, nudge_key="draft")

    def run(self, question: str, use_retrieval: bool = True) -> dict:
        facts = snapshot(self.df)
        hits = self.rag.search_guidance(question + " overspending savings", n=4) if (self.rag and use_retrieval) else []
        guidance = "\n".join(f"- [{h['section']}] {h['text']}" for h in hits) or "(none retrieved)"
        draft = self.advise_chain.invoke({"question": question, "facts": facts, "guidance": guidance})
        flagged = unsupported_numbers(draft.answer, facts, guidance)
        answer, reflected, claims = draft.answer, False, []
        if flagged:  # reflection pass
            fixed = self.reflect_chain.invoke({"facts": facts, "guidance": guidance, "draft": draft.answer,
                                               "flagged": ", ".join(f"{x:,.2f}" for x in flagged)})
            answer, reflected, claims = fixed.revised_answer, True, fixed.unsupported_claims
        return {"answer": answer, "reasoning": draft.reasoning, "guidance": hits, "facts": facts,
                "reflected": reflected, "flagged_numbers": flagged, "unsupported_claims": claims}
