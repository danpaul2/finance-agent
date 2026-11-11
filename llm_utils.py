"""Robust structured-output helpers shared by every chain.

RobustPydanticParser : tolerant of <think> blocks and prose around the JSON; on failure the error
                       says what the model actually returned (or that it was empty, and why).
with_repair          : on an unparsable reply, re-asks up to 3 times with a stricter instruction.
                       API errors (rate limits, auth) are NOT retried here, so no quota is burned.
"""
from __future__ import annotations
import re
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import Generation
from langchain_core.runnables import RunnableLambda


def _first_json_object(text: str) -> str | None:
    """First balanced {...} in text (string-aware), or None if unbalanced/missing."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


class RobustPydanticParser(PydanticOutputParser):
    def parse_result(self, result, *, partial: bool = False):
        gen = result[0]
        raw = getattr(gen, "text", "") or ""
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I).strip()
        if not text:
            msg = getattr(gen, "message", None)
            meta = getattr(msg, "response_metadata", None) or {}
            extra = getattr(msg, "additional_kwargs", None) or {}
            reasoning = str(extra.get("reasoning_content") or "")
            # last resort: some reasoning models put the final JSON in the reasoning channel
            blob = _first_json_object(reasoning) if reasoning else None
            if blob:
                try:
                    return super().parse_result([Generation(text=blob)], partial=partial)
                except OutputParserException:
                    pass
            raise OutputParserException(
                f"The model returned an empty reply (finish_reason={meta.get('finish_reason')}, "
                f"reasoning_chars={len(reasoning)}, extra_fields={sorted(extra)}).", llm_output=raw)
        blob = _first_json_object(text) or text
        try:
            return super().parse_result([Generation(text=blob)], partial=partial)
        except OutputParserException as e:
            raise OutputParserException(f"{e}\nModel output was: {text[:300]!r}", llm_output=raw) from e


def with_repair(chain, nudge_key: str, attempts: int = 3):
    """Wrap a prompt|llm|parser chain: retry only on unparsable output, nudging via `nudge_key`."""
    def run(inputs: dict):
        err = None
        for i in range(attempts):
            x = dict(inputs)
            if i:
                x[nudge_key] = f"{inputs[nudge_key]}\n\nIMPORTANT: reply with ONLY one valid JSON object and nothing else."
            try:
                return chain.invoke(x)
            except OutputParserException as e:
                err = e
        raise err
    return RunnableLambda(run)
