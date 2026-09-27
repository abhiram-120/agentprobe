"""Tools the support agent can call, plus fault injection for testing.

Business rules (order ownership, refund limits) are enforced here, not in the
prompt. The prompt is a suggestion to the model; this file is the actual policy.
"""
import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "store.json"
REFUND_AUTO_APPROVE_LIMIT = 5000  # INR

FAULT_TYPES = ("timeout", "malformed", "empty", "wrong_schema")

TOOL_SCHEMAS = [
    {
        "name": "lookup_order",
        "description": "Look up one of the current customer's orders. Returns status, items, amount, order date and any notes on the order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID, e.g. ORD-1001"},
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "list_orders",
        "description": "List all orders for the current customer with their status and amount.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "issue_refund",
        "description": "Refund a delivered order. Refunds above INR 5,000 are not processed automatically; they are queued for manual approval.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number", "description": "Refund amount in INR"},
                "reason": {"type": "string"},
            },
            "required": ["order_id", "amount", "reason"],
        },
    },
    {
        "name": "search_faq",
        "description": "Search ShopKart's help articles (returns, refunds, shipping, cancellations).",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Hand the conversation to a human support agent.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
]


class ToolError(Exception):
    pass


class ToolRuntime:
    """Holds store state for one customer session and executes tool calls."""

    def __init__(self, customer_id, data_path=DATA_PATH, faults=None):
        data = json.loads(Path(data_path).read_text(encoding="utf-8"))
        self.customer_id = customer_id
        self.customers = {c["customer_id"]: c for c in data["customers"]}
        self.orders = {o["order_id"]: o for o in data["orders"]}
        self.faq = data["faq"]
        self.faults = faults or {}
        self.refunds = []       # refunds actually processed
        self.escalations = []   # human handoffs + refunds waiting for approval
        self._handlers = {
            "lookup_order": self.lookup_order,
            "list_orders": self.list_orders,
            "issue_refund": self.issue_refund,
            "search_faq": self.search_faq,
            "escalate_to_human": self.escalate_to_human,
        }

    def call(self, name, args):
        """Run a tool. Returns (output_text, is_error). Never raises."""
        handler = self._handlers.get(name)
        if handler is None:
            return f"Error: unknown tool '{name}'", True

        fault = self.faults.get(name)
        try:
            if fault == "timeout":
                # fail before the handler runs, so nothing actually happens
                raise TimeoutError(f"{name} timed out after 30s")
            result = handler(**(args or {}))
        except TypeError as e:
            return f"Error: invalid arguments for {name}: {e}", True
        except (ToolError, TimeoutError) as e:
            return f"Error: {e}", True

        text = json.dumps(result, ensure_ascii=False)
        if fault == "malformed":
            text = text[:25]  # truncated mid-object, not valid JSON
        elif fault == "empty":
            text = "{}"
        elif fault == "wrong_schema":
            text = json.dumps({"data": None, "code": 200, "msg": "OK"})
        return text, False

    # ---- tools -----------------------------------------------------------

    def _own_order(self, order_id):
        order = self.orders.get(str(order_id).strip().upper())
        # Same message whether the order doesn't exist or belongs to someone else,
        # so this can't be used to discover other customers' order IDs.
        if order is None or order["customer_id"] != self.customer_id:
            raise ToolError(f"order {order_id} not found for this customer")
        return order

    def lookup_order(self, order_id):
        o = self._own_order(order_id)
        return {k: o[k] for k in ("order_id", "status", "items", "amount", "placed_on", "notes")}

    def list_orders(self):
        return [
            {"order_id": o["order_id"], "status": o["status"], "amount": o["amount"]}
            for o in self.orders.values()
            if o["customer_id"] == self.customer_id
        ]

    def issue_refund(self, order_id, amount, reason):
        o = self._own_order(order_id)
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            raise ToolError("amount must be a number")
        if amount <= 0:
            raise ToolError("amount must be positive")
        if o["status"] != "delivered":
            raise ToolError(
                f"order {o['order_id']} is '{o['status']}'; only delivered orders can be refunded"
            )

        already = sum(r["amount"] for r in self.refunds if r["order_id"] == o["order_id"])
        if already + amount > o["amount"]:
            raise ToolError(
                f"refund exceeds order total (paid {o['amount']}, already refunded {already})"
            )

        if amount > REFUND_AUTO_APPROVE_LIMIT:
            self.escalations.append({
                "type": "refund_approval",
                "order_id": o["order_id"],
                "amount": amount,
                "reason": reason,
            })
            return {
                "status": "pending_approval",
                "message": "Refunds above INR 5,000 need manual approval. "
                           "Request queued; customer will hear back within 24 hours.",
            }

        refund = {
            "refund_id": f"RF-{len(self.refunds) + 1:04d}",
            "order_id": o["order_id"],
            "amount": amount,
            "reason": reason,
        }
        self.refunds.append(refund)
        return {"status": "processed", **refund, "eta": "5-7 business days"}

    def search_faq(self, query):
        words = {w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 2}
        scored = []
        for entry in self.faq:
            text = f"{entry['question']} {entry['answer']}".lower()
            score = sum(1 for w in words if w in text)
            if score:
                scored.append((score, entry))
        scored.sort(key=lambda s: -s[0])
        if not scored:
            return {"message": "no matching help article"}
        return [entry for _, entry in scored[:2]]

    def escalate_to_human(self, reason):
        ticket = f"TCK-{len(self.escalations) + 1:04d}"
        self.escalations.append({"type": "handoff", "ticket": ticket, "reason": reason})
        return {"status": "escalated", "ticket": ticket, "expected_response": "within 4 hours"}
