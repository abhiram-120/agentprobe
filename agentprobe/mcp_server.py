"""Exposes the ShopKart tools over MCP (stdio).

Try it with MCP Inspector:
    npx @modelcontextprotocol/inspector python -m agentprobe.mcp_server
"""
import os

from mcp.server.fastmcp import FastMCP

from .tools import ToolRuntime

mcp = FastMCP("shopkart-support")
runtime = ToolRuntime(customer_id=os.getenv("AGENTPROBE_CUSTOMER", "C001"))


def _run(name, **args):
    output, is_error = runtime.call(name, args)
    return output


@mcp.tool()
def lookup_order(order_id: str) -> str:
    """Look up one of the current customer's orders."""
    return _run("lookup_order", order_id=order_id)


@mcp.tool()
def list_orders() -> str:
    """List all orders for the current customer."""
    return _run("list_orders")


@mcp.tool()
def issue_refund(order_id: str, amount: float, reason: str) -> str:
    """Refund a delivered order. Above INR 5,000 goes to manual approval."""
    return _run("issue_refund", order_id=order_id, amount=amount, reason=reason)


@mcp.tool()
def search_faq(query: str) -> str:
    """Search ShopKart help articles."""
    return _run("search_faq", query=query)


@mcp.tool()
def escalate_to_human(reason: str) -> str:
    """Hand the conversation to a human agent."""
    return _run("escalate_to_human", reason=reason)


if __name__ == "__main__":
    mcp.run()
