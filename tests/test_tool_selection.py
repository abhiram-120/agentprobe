import pytest

CASES = [
    ("Where is my order ORD-1002?", "lookup_order", {"issue_refund"}),
    ("What's your return policy?", "search_faq", {"issue_refund"}),
    ("Can you show me all my orders?", "list_orders", {"issue_refund"}),
    ("I want to talk to a real person please.", "escalate_to_human", {"issue_refund"}),
]


@pytest.mark.llm
@pytest.mark.parametrize("message,expected,forbidden", CASES)
def test_picks_right_tool(make_agent, message, expected, forbidden):
    result = make_agent().run(message)
    names = [c.name for c in result.tool_calls]
    assert expected in names, f"expected {expected}, got {names} (run {result.run_id})"
    assert not forbidden & set(names), f"called forbidden tool: {names} (run {result.run_id})"


@pytest.mark.llm
def test_refund_arguments_match_order(make_agent, runtime):
    result = make_agent().run("The earbuds from ORD-1001 stopped charging after a week. Please refund me.")
    calls = result.called("issue_refund")
    assert calls, f"no refund attempted (run {result.run_id})"
    args = calls[-1].input
    assert args["order_id"].strip().upper() == "ORD-1001"
    assert float(args["amount"]) == 1299
    assert len(runtime.refunds) == 1
