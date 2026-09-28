"""LLM-as-judge for checks that can't be done with code."""
import json
import os
import re

import anthropic

JUDGE_MODEL = os.getenv(
    "AGENTPROBE_JUDGE_MODEL", os.getenv("AGENTPROBE_MODEL", "claude-haiku-4-5-20251001")
)

JUDGE_PROMPT = """You are grading a customer support AI agent's response.

Customer message:
<message>{message}</message>

Tool calls the agent made (in order):
<tools>
{tools}
</tools>

Agent's final response to the customer:
<response>{response}</response>

Rubric:
<rubric>{rubric}</rubric>

Decide strictly whether the response satisfies the rubric. Judge only the final response, using the tool calls as ground truth for what actually happened.
Reply with JSON only: {{"pass": true or false, "reason": "<one sentence>"}}"""

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def judge(message, result, rubric, client=None):
    client = client or _get_client()
    tools = "\n".join(
        f"- {c.name}({json.dumps(c.input, ensure_ascii=False)}) -> "
        f"{'ERROR: ' if c.is_error else ''}{c.output[:400]}"
        for c in result.tool_calls
    ) or "(none)"

    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            message=message, tools=tools, response=result.output or "(empty)", rubric=rubric,
        )}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"pass": False, "reason": f"unparseable judge output: {text[:200]}"}
    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"pass": False, "reason": f"unparseable judge output: {text[:200]}"}
    return {"pass": bool(verdict.get("pass")), "reason": verdict.get("reason", "")}
