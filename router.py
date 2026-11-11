"""Intent detection + routing (criterion 6) and Agentic RAG orchestration (criterion 5).

Pipeline (all LCEL):   RunnableLambda(route)  |  RunnableBranch(handlers...)

  transaction_lookup -> ChromaDB similarity search, LLM picks relevant rows, CODE sums them
  spending_analysis  -> ReAct loop over pandas/calculator tools; the agent may also call the
                        search_transactions / search_guidance tools when it decides retrieval helps
  savings_guidance   -> exact facts + (if the router says needs_retrieval) retrieved guidance -> Advisor (+ reflection)
  out_of_scope       -> short canned reply, no further LLM call
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llm_utils import RobustPydanticParser, with_repair
from langchain_core.runnables import RunnableBranch, RunnableLambda

from prompts import (ROUTER_SYSTEM, ROUTER_USER, ROUTER_FEWSHOT_MESSAGES, REACT_SYSTEM, REACT_USER,
                     LOOKUP_SYSTEM, LOOKUP_USER)
from agents import analysis as A
from agents.advisor import Advisor
from agents.grounded import (answer_usable, grounded_analysis, grounded_guidance,
                             looks_like_refusal, matching_transactions,
                             mentioned_categories, mentioned_merchant)

MAX_REACT_STEPS = 4


# ---------------- schemas ----------------
class RouteDecision(BaseModel):
    reason: str = Field(description="max 15 words, written before the intent")
    intent: Literal["transaction_lookup", "spending_analysis", "savings_guidance", "out_of_scope"]
    needs_retrieval: bool


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ReActStep(BaseModel):
    thought: str
    actions: list[ToolCall] = Field(default_factory=list)
    final_answer: Optional[str] = None


class LookupAnswer(BaseModel):
    relevant_ids: list[str] = Field(default_factory=list)
    answer: str


@dataclass
class AgentResult:
    question: str
    intent: str
    answer: str
    route_source: str = "llm"
    trace: list[str] = field(default_factory=list)       # human-readable steps for the UI
    tables: dict[str, Any] = field(default_factory=dict)  # e.g. matched transactions
    sources: list[dict] = field(default_factory=list)     # retrieved guidance passages


# ---------------- rule shortcuts (save an LLM call on unambiguous requests) ----------------
_GREETING = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|bye)\b[\s!.]*$", re.I)
_ADVICE = re.compile(r"\b(how (can|do|should) i (save|cut|reduce|spend less)|where (am i|can i) (overspend|cut|save|reduce)|"
                     r"budget|savings? (tips|advice|plan)|save more|overspending)\b", re.I)


def shortcut_route(q: str) -> RouteDecision | None:
    if _GREETING.match(q):
        return RouteDecision(reason="greeting or thanks", intent="out_of_scope", needs_retrieval=False)
    if _ADVICE.search(q):
        return RouteDecision(reason="explicit request for savings advice", intent="savings_guidance", needs_retrieval=True)
    return None


# ---------------- chain builders ----------------
def _structured_chain(llm, system, user, model, fewshot=(), nudge_key="question"):
    parser = RobustPydanticParser(pydantic_object=model)
    prompt = ChatPromptTemplate.from_messages([("system", system), *fewshot, ("human", user)]).partial(
        format_instructions=parser.get_format_instructions())
    return with_repair(prompt | llm | parser, nudge_key)


def inr(x: float) -> str:
    return f"\u20b9{abs(x):,.2f}"


class FinanceAgent:
    def __init__(self, df, llm, rag, use_shortcuts: bool = True):
        self.df, self.rag, self.use_shortcuts = df, rag, use_shortcuts
        self.tools = A.make_tools(df, rag)
        self.router_chain = _structured_chain(llm, ROUTER_SYSTEM, ROUTER_USER, RouteDecision, ROUTER_FEWSHOT_MESSAGES)
        self.react_chain = _structured_chain(llm, REACT_SYSTEM, REACT_USER, ReActStep)
        self.lookup_chain = _structured_chain(llm, LOOKUP_SYSTEM, LOOKUP_USER, LookupAnswer)
        self.advisor = Advisor(llm, df, rag)
        self.info = A.data_info(df)

        branch = RunnableBranch(
            (lambda s: s["intent"] == "transaction_lookup", RunnableLambda(self._lookup)),
            (lambda s: s["intent"] == "spending_analysis", RunnableLambda(self._analysis)),
            (lambda s: s["intent"] == "savings_guidance", RunnableLambda(self._guidance)),
            RunnableLambda(self._out_of_scope),
        )
        self.pipeline = RunnableLambda(self._route) | branch

    # ---- public ----
    def ask(self, question: str) -> AgentResult:
        s = self.pipeline.invoke({"question": question, "trace": []})
        return AgentResult(question=question, intent=s["intent"], answer=s["answer"], route_source=s["route_source"],
                           trace=s["trace"], tables=s.get("tables", {}), sources=s.get("sources", []))

    # ---- step 1: intent detection ----
    def _route(self, s: dict) -> dict:
        d = shortcut_route(s["question"]) if self.use_shortcuts else None
        src = "rule"
        if d is None:
            d, src = self.router_chain.invoke({"question": s["question"]}), "llm"
        s.update(intent=d.intent, needs_retrieval=d.needs_retrieval, route_source=src)
        s["trace"].append(f"Router ({src}): intent={d.intent}, retrieval={'yes' if d.needs_retrieval else 'no'} - {d.reason}")
        return s

    # ---- handlers ----
    def _lookup(self, s: dict) -> dict:
        hits = self.rag.search_transactions(s["question"], n=15)
        s["trace"].append(f"ChromaDB: retrieved {len(hits)} candidate transactions")
        direct = matching_transactions(self.df, s["question"])
        if direct:
            s["trace"].append(f"Statement filter: {len(direct)} rows match the merchant, category, or keyword")
        seen = {h["id"] for h in hits}
        for row in direct:
            if row["id"] not in seen:
                hits.append(row)
                seen.add(row["id"])
        rows = "\n".join(f"{h['id']} | {h['date']} | {h['merchant']} | {h['description']} | {h['category']} | {h['amount']:+,.2f}"
                         for h in hits)
        try:
            out = self.lookup_chain.invoke({"question": s["question"], "rows": rows or "(none)"})
            chosen = [h for h in hits if h["id"] in set(out.relevant_ids)]
            answer = out.answer
        except Exception as e:  # noqa: BLE001
            chosen, answer = [], ""
            s["trace"].append(f"Lookup model unavailable ({type(e).__name__}); using the statement filter")
        precise = bool(mentioned_merchant(self.df, s["question"]) or mentioned_categories(self.df, s["question"]))
        if direct and (not chosen or looks_like_refusal(answer) or precise):
            chosen = direct
            if not answer or looks_like_refusal(answer):
                answer = "Here are the matching transactions from your statement."
        if chosen:
            chosen = sorted(chosen, key=lambda h: h["date"])
            total = sum(h["amount"] for h in chosen)  # exact, computed by code
            table = ["| Date | Merchant | Description | Category | Amount |", "|---|---|---|---|---:|"]
            table += [f"| {h['date']} | {h['merchant']} | {h['description']} | {h['category']} | "
                      f"{'-' if h['amount'] < 0 else '+'}{inr(h['amount'])} |" for h in chosen]  # rows straight from the data
            answer += "\n\n" + "\n".join(table)
            answer += f"\n\n**{len(chosen)} matching transactions, net {'-' if total < 0 else '+'}{inr(total)}** (computed by code)."
        s["trace"].append(f"Selected {len(chosen)} relevant rows; total computed by code")
        s.update(answer=answer, tables={"transactions": chosen})
        return s

    def _analysis(self, s: dict) -> dict:
        scratch = ""
        try:
            for step in range(MAX_REACT_STEPS):
                last = step == MAX_REACT_STEPS - 1
                note = "This is your last step: you MUST provide final_answer now using the observations so far." if last \
                    else "Only set final_answer once the observations contain every number you need."
                out = self.react_chain.invoke({"question": s["question"], "tools": A.tool_descriptions(self.tools),
                                               "data_info": self.info, "step_note": note, "scratchpad": scratch or "(none yet)"})
                if answer_usable(out.final_answer, scratch):
                    s["trace"].append(f"ReAct step {step + 1}: final answer")
                    s["answer"] = out.final_answer.strip()
                    return s
                if not out.actions:
                    s["trace"].append(f"ReAct step {step + 1}: no usable answer")
                    break
                for a in out.actions[:3]:
                    obs = A.run_tool(self.tools, a.tool, a.args)
                    s["trace"].append(f"ReAct step {step + 1}: {a.tool}({a.args})")
                    scratch += f"Thought: {out.thought}\nAction: {a.tool}({a.args})\nObservation: {obs}\n\n"
        except Exception as e:  # noqa: BLE001
            s["trace"].append(f"ReAct stopped: {type(e).__name__}: {str(e)[:180]}")
        s["trace"].append("Answer computed from the statement tools")
        s["answer"] = grounded_analysis(self.df, s["question"])
        return s

    def _guidance(self, s: dict) -> dict:
        try:
            r = self.advisor.run(s["question"], use_retrieval=s["needs_retrieval"])
        except Exception as e:  # noqa: BLE001
            s["trace"].append(f"Advisor model unavailable ({type(e).__name__}); answering from computed facts")
            hits = self.rag.search_guidance(s["question"] + " overspending savings", n=3) if (self.rag and s["needs_retrieval"]) else []
            s["trace"].append("Facts computed by code (pandas)")
            s.update(answer=grounded_guidance(self.df, s["question"], hits), sources=hits)
            return s
        if looks_like_refusal(r["answer"]):
            s["trace"].append("Advisor declined; answering from computed facts")
            s.update(answer=grounded_guidance(self.df, s["question"], r["guidance"]), sources=r["guidance"],
                     reasoning=r.get("reasoning", ""))
            return s
        s["trace"].append("Facts computed by code (pandas)")
        if s["needs_retrieval"]:
            s["trace"].append(f"ChromaDB: retrieved {len(r['guidance'])} guidance passages")
        s["trace"].append("Reflection: " + (f"revised draft (unsupported numbers: {r['flagged_numbers']})" if r["reflected"]
                                            else "all numbers verified against facts"))
        s.update(answer=r["answer"], sources=r["guidance"], reasoning=r["reasoning"])
        return s

    def _out_of_scope(self, s: dict) -> dict:
        s["answer"] = ("I can help with your statement: look up transactions, analyse spending (totals, trends, categories) "
                       "or suggest ways to save. Try, for example, \"How much did I spend on food in September?\".")
        return s
