"""LLM-as-judge for checks that can't be done with code."""
import json
import os
import re

JUDGE_MODEL = os.getenv(
    "AGENTPROBE_JUDGE_MODEL",
    os.getenv(
        "AGENTPROBE_MODEL",
        "gemini-2.5-flash" if (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        else "claude-haiku-4-5-20251001",
    ),
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


def _generate(prompt: str) -> str:
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or JUDGE_MODEL.startswith("gemini"):
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=300),
        )
        return (resp.text or "").strip()

    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def judge(message, result, rubric, client=None):
    tools = "\n".join(
        f"- {c.name}({json.dumps(c.input, ensure_ascii=False)}) -> "
        f"{'ERROR: ' if c.is_error else ''}{c.output[:400]}"
        for c in result.tool_calls
    ) or "(none)"

    text = _generate(JUDGE_PROMPT.format(
        message=message, tools=tools, response=result.output or "(empty)", rubric=rubric,
    ))
    # Models often wrap JSON in ```json fences; strip that before parsing.
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {"pass": False, "reason": f"unparseable judge output: {text[:200]}"}
    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"pass": False, "reason": f"unparseable judge output: {text[:200]}"}
    return {"pass": bool(verdict.get("pass")), "reason": verdict.get("reason", "")}
