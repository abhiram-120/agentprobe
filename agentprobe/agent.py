"""A small tool-calling support agent built directly on the Anthropic Messages API.

No framework on purpose: every step of the loop is visible and gets traced.
"""
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from .tools import TOOL_SCHEMAS, ToolRuntime
from .tracer import Tracer

DEFAULT_MODEL = os.getenv("AGENTPROBE_MODEL", "claude-haiku-4-5-20251001")
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


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


class SupportAgent:
    def __init__(self, runtime: ToolRuntime, prompt_version="v2", model=DEFAULT_MODEL,
                 tracer=None, max_steps=6, client=None):
        self.runtime = runtime
        self.prompt_version = prompt_version
        self.model = model
        self.system = (PROMPTS_DIR / f"{prompt_version}.txt").read_text(encoding="utf-8")
        self.tracer = tracer or Tracer()
        self.max_steps = max_steps
        self.client = client or anthropic.Anthropic()

    def run(self, user_message: str) -> AgentResult:
        run_id = self.tracer.start_run(self.model, self.prompt_version, user_message)
        messages = [{"role": "user", "content": user_message}]
        calls = []
        tokens_in = tokens_out = 0
        started = time.monotonic()

        for _ in range(self.max_steps):
            resp = self.client.messages.create(
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

        # ran out of steps without a final answer
        return self._finish(run_id, "", "max_steps", calls, started, tokens_in, tokens_out)

    def _finish(self, run_id, text, status, calls, started, tokens_in, tokens_out):
        latency_ms = int((time.monotonic() - started) * 1000)
        self.tracer.end_run(run_id, text, status, latency_ms, tokens_in, tokens_out)
        return AgentResult(run_id=run_id, output=text, status=status, tool_calls=calls)
