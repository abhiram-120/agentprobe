"""A small tool-calling support agent.

Backends: local heuristic, Groq (OpenAI-compatible), Google Gemini, or Anthropic.
No framework on purpose: every step of the loop is visible and gets traced.
"""
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .tools import TOOL_SCHEMAS, ToolRuntime
from .tracer import Tracer

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _default_model():
    backend = os.getenv("AGENTPROBE_BACKEND", "").lower()
    if backend == "local" or os.getenv("AGENTPROBE_MODEL", "").startswith("local"):
        return os.getenv("AGENTPROBE_MODEL", "local-heuristic")
    if backend == "groq" or os.getenv("GROQ_API_KEY"):
        return os.getenv("AGENTPROBE_MODEL", "openai/gpt-oss-20b")
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return os.getenv("AGENTPROBE_MODEL", "gemini-2.5-flash")
    if os.getenv("ANTHROPIC_API_KEY"):
        return os.getenv("AGENTPROBE_MODEL", "claude-haiku-4-5-20251001")
    return os.getenv("AGENTPROBE_MODEL", "local-heuristic")


DEFAULT_MODEL = _default_model()


def _pick_backend(model: str) -> str:
    forced = os.getenv("AGENTPROBE_BACKEND", "").lower()
    if forced in ("local", "gemini", "anthropic", "groq"):
        return forced
    if model.startswith("local"):
        return "local"
    if os.getenv("GROQ_API_KEY") or model.startswith(("openai/", "qwen/", "meta-llama/")):
        return "groq"
    if model.startswith("gemini") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        if not model.startswith("claude"):
            return "gemini"
    return "anthropic"


@dataclass
class ToolCall:
    name: str
    input: dict
    output: str
    is_error: bool


@dataclass
class AgentResult:
    run_id: str
    output: str
    status: str  # "ok" or "max_steps"
    tool_calls: list = field(default_factory=list)

    def called(self, name):
        return [c for c in self.tool_calls if c.name == name]


def _openai_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema.get("description", ""),
                "parameters": schema.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for schema in TOOL_SCHEMAS
    ]


def _gemini_tools():
    from google.genai import types

    decls = []
    for schema in TOOL_SCHEMAS:
        decls.append(types.FunctionDeclaration(
            name=schema["name"],
            description=schema.get("description", ""),
            parameters=schema.get("input_schema") or {"type": "object", "properties": {}},
        ))
    return [types.Tool(function_declarations=decls)]


class SupportAgent:
    def __init__(self, runtime: ToolRuntime, prompt_version="v2", model=None,
                 tracer=None, max_steps=6, client=None):
        self.runtime = runtime
        self.prompt_version = prompt_version
        self.model = model or DEFAULT_MODEL
        self.system = (PROMPTS_DIR / f"{prompt_version}.txt").read_text(encoding="utf-8")
        self.tracer = tracer or Tracer()
        self.max_steps = max_steps
        self.client = client
        self._backend = _pick_backend(self.model)

    def run(self, user_message: str) -> AgentResult:
        if self._backend == "local":
            from .local_agent import LocalSupportAgent
            return LocalSupportAgent(
                self.runtime,
                prompt_version=self.prompt_version,
                model=self.model if self.model.startswith("local") else "local-heuristic",
                tracer=self.tracer,
                max_steps=self.max_steps,
            ).run(user_message)
        if self._backend == "groq":
            return self._run_groq(user_message)
        if self._backend == "gemini":
            return self._run_gemini(user_message)
        return self._run_anthropic(user_message)

    def _run_groq(self, user_message: str) -> AgentResult:
        from openai import OpenAI

        client = self.client or OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        run_id = self.tracer.start_run(self.model, self.prompt_version, user_message)
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user_message},
        ]
        calls = []
        tokens_in = tokens_out = 0
        started = time.monotonic()
        tools = _openai_tools()

        for _ in range(self.max_steps):
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=0,
                max_tokens=1024,
            )
            usage = resp.usage
            if usage:
                tokens_in += int(usage.prompt_tokens or 0)
                tokens_out += int(usage.completion_tokens or 0)

            msg = resp.choices[0].message
            tool_calls = msg.tool_calls or []
            self.tracer.log_step(run_id, "model", {
                "stop_reason": "tool_use" if tool_calls else "end_turn",
                "content": (
                    [
                        {
                            "type": "tool_use",
                            "name": tc.function.name,
                            "input": json.loads(tc.function.arguments or "{}"),
                        }
                        for tc in tool_calls
                    ]
                    if tool_calls
                    else [{"type": "text", "text": msg.content or ""}]
                ),
            })

            assistant_msg = {"role": "assistant", "content": msg.content}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)

            if not tool_calls:
                return self._finish(
                    run_id, (msg.content or "").strip(), "ok", calls, started, tokens_in, tokens_out
                )

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                output, is_error = self.runtime.call(name, args)
                calls.append(ToolCall(name, dict(args), output, is_error))
                self.tracer.log_step(run_id, "tool", {
                    "name": name,
                    "input": args,
                    "output": output,
                    "is_error": is_error,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                })

        return self._finish(run_id, "", "max_steps", calls, started, tokens_in, tokens_out)

    def _run_gemini(self, user_message: str) -> AgentResult:
        from google import genai
        from google.genai import types

        client = self.client or genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        run_id = self.tracer.start_run(self.model, self.prompt_version, user_message)
        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        ]
        calls = []
        tokens_in = tokens_out = 0
        started = time.monotonic()
        config = types.GenerateContentConfig(
            system_instruction=self.system,
            tools=_gemini_tools(),
            temperature=0,
            max_output_tokens=1024,
        )

        for _ in range(self.max_steps):
            resp = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            usage = getattr(resp, "usage_metadata", None)
            if usage:
                tokens_in += int(getattr(usage, "prompt_token_count", 0) or 0)
                tokens_out += int(getattr(usage, "candidates_token_count", 0) or 0)

            candidate = (resp.candidates or [None])[0]
            if candidate is None or candidate.content is None:
                text = (resp.text or "").strip()
                return self._finish(run_id, text, "ok", calls, started, tokens_in, tokens_out)

            parts = candidate.content.parts or []
            fn_calls = [p for p in parts if getattr(p, "function_call", None)]
            text_parts = [p.text for p in parts if getattr(p, "text", None)]

            self.tracer.log_step(run_id, "model", {
                "stop_reason": "tool_use" if fn_calls else "end_turn",
                "content": [
                    {"type": "tool_use", "name": p.function_call.name,
                     "input": dict(p.function_call.args or {})}
                    if getattr(p, "function_call", None)
                    else {"type": "text", "text": p.text or ""}
                    for p in parts
                ],
            })
            contents.append(candidate.content)

            if not fn_calls:
                text = "".join(text_parts).strip()
                return self._finish(run_id, text, "ok", calls, started, tokens_in, tokens_out)

            result_parts = []
            for p in fn_calls:
                name = p.function_call.name
                args = dict(p.function_call.args or {})
                output, is_error = self.runtime.call(name, args)
                calls.append(ToolCall(name, args, output, is_error))
                self.tracer.log_step(run_id, "tool", {
                    "name": name,
                    "input": args,
                    "output": output,
                    "is_error": is_error,
                })
                payload = {"result": output, "is_error": is_error}
                result_parts.append(
                    types.Part.from_function_response(name=name, response=payload)
                )
            contents.append(types.Content(role="user", parts=result_parts))

        return self._finish(run_id, "", "max_steps", calls, started, tokens_in, tokens_out)

    def _run_anthropic(self, user_message: str) -> AgentResult:
        import anthropic

        client = self.client or anthropic.Anthropic()
        run_id = self.tracer.start_run(self.model, self.prompt_version, user_message)
        messages = [{"role": "user", "content": user_message}]
        calls = []
        tokens_in = tokens_out = 0
        started = time.monotonic()

        for _ in range(self.max_steps):
            resp = client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0,
                system=self.system,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            tokens_in += resp.usage.input_tokens
            tokens_out += resp.usage.output_tokens
            self.tracer.log_step(run_id, "model", {
                "stop_reason": resp.stop_reason,
                "content": [b.model_dump() for b in resp.content],
            })
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text").strip()
                return self._finish(run_id, text, "ok", calls, started, tokens_in, tokens_out)

            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                output, is_error = self.runtime.call(block.name, block.input)
                calls.append(ToolCall(block.name, dict(block.input), output, is_error))
                self.tracer.log_step(run_id, "tool", {
                    "name": block.name,
                    "input": block.input,
                    "output": output,
                    "is_error": is_error,
                })
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                })
            messages.append({"role": "user", "content": results})

        return self._finish(run_id, "", "max_steps", calls, started, tokens_in, tokens_out)

    def _finish(self, run_id, text, status, calls, started, tokens_in, tokens_out):
        latency_ms = int((time.monotonic() - started) * 1000)
        self.tracer.end_run(run_id, text, status, latency_ms, tokens_in, tokens_out)
        return AgentResult(run_id=run_id, output=text, status=status, tool_calls=calls)
