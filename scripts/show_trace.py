"""Print every step of an agent run.

    python scripts/show_trace.py last
    python scripts/show_trace.py 3f9a1c0b2d7e
"""
import json
import os
import sqlite3
import sys


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: show_trace.py <run_id|last>")
    conn = sqlite3.connect(os.getenv("AGENTPROBE_TRACE_DB", "traces.db"))
    run_id = sys.argv[1]
    if run_id == "last":
        row = conn.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
        if not row:
            sys.exit("no runs in trace db")
        run_id = row[0]

    run = conn.execute(
        "SELECT model, prompt_version, user_input, final_output, status, latency_ms, "
        "input_tokens, output_tokens FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if not run:
        sys.exit(f"run {run_id} not found")
    model, prompt, user_input, output, status, latency, t_in, t_out = run

    print(f"run {run_id}  model={model}  prompt={prompt}  status={status}  "
          f"{latency}ms  tokens={t_in}/{t_out}")
    print(f"\nUSER: {user_input}\n")

    for step_no, kind, payload in conn.execute(
        "SELECT step_no, kind, payload FROM steps WHERE run_id = ? ORDER BY step_no", (run_id,)
    ):
        data = json.loads(payload)
        if kind == "model":
            for block in data["content"]:
                if block.get("type") == "text" and block.get("text", "").strip():
                    print(f"[{step_no}] MODEL: {block['text'].strip()}")
                elif block.get("type") == "tool_use":
                    print(f"[{step_no}] MODEL -> {block['name']}({json.dumps(block['input'], ensure_ascii=False)})")
        elif kind == "tool":
            tag = "TOOL ERROR" if data["is_error"] else "TOOL"
            out = data["output"]
            print(f"[{step_no}] {tag} <- {out[:300]}{'...' if len(out) > 300 else ''}")

    print(f"\nFINAL: {output}")


if __name__ == "__main__":
    main()
