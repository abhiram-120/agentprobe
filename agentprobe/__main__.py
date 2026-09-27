"""Send a single message to the support agent from the command line."""
import argparse
import json

from .agent import SupportAgent
from .tools import ToolRuntime


def main():
    p = argparse.ArgumentParser(description="Send one message to the ShopKart support agent.")
    p.add_argument("message")
    p.add_argument("--customer", default="C001")
    p.add_argument("--prompt", default="v2", help="prompt version in prompts/")
    p.add_argument("--fault", action="append", default=[],
                   help="inject a tool fault, e.g. lookup_order=timeout")
    args = p.parse_args()

    faults = dict(f.split("=", 1) for f in args.fault)
    agent = SupportAgent(ToolRuntime(args.customer, faults=faults), prompt_version=args.prompt)
    result = agent.run(args.message)

    for c in result.tool_calls:
        flag = "  [error]" if c.is_error else ""
        print(f"  -> {c.name}({json.dumps(c.input, ensure_ascii=False)}){flag}")
    print()
    print(result.output or f"(no output, status={result.status})")
    print(f"\nrun_id: {result.run_id}")


if __name__ == "__main__":
    main()
