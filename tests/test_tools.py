"""Tool layer and business rules. No model calls."""
import json

import pytest

from agentprobe.tools import ToolRuntime


@pytest.fixture
def rt():
    return ToolRuntime(customer_id="C001")


def test_lookup_own_order(rt):
    out, err = rt.call("lookup_order", {"order_id": "ORD-1001"})
    assert not err
    data = json.loads(out)
    assert data["status"] == "delivered"
    assert data["amount"] == 1299


def test_lookup_is_case_and_whitespace_tolerant(rt):
    out, err = rt.call("lookup_order", {"order_id": "  ord-1001 "})
    assert not err


def test_other_customers_order_looks_like_missing_order(rt):
    foreign, err1 = rt.call("lookup_order", {"order_id": "ORD-1003"})
    missing, err2 = rt.call("lookup_order", {"order_id": "ORD-9999"})
    assert err1 and err2
    # same shape of error, so the tool can't be used to probe which IDs exist
    assert foreign.replace("ORD-1003", "X") == missing.replace("ORD-9999", "X")
    assert "Priya" not in foreign


def test_list_orders_only_returns_own(rt):
    out, _ = rt.call("list_orders", {})
    ids = {o["order_id"] for o in json.loads(out)}
    assert ids == {"ORD-1001", "ORD-1002", "ORD-1004", "ORD-1005", "ORD-1006"}


def test_small_refund_processed(rt):
    out, err = rt.call("issue_refund", {"order_id": "ORD-1001", "amount": 1299, "reason": "defective"})
    assert not err
    assert json.loads(out)["status"] == "processed"
    assert len(rt.refunds) == 1


def test_refund_over_limit_goes_to_approval(rt):
    out, err = rt.call("issue_refund", {"order_id": "ORD-1005", "amount": 12999, "reason": "cracked"})
    assert not err
    assert json.loads(out)["status"] == "pending_approval"
    assert rt.refunds == []
    assert rt.escalations[0]["type"] == "refund_approval"


def test_refund_on_shipped_order_rejected(rt):
    out, err = rt.call("issue_refund", {"order_id": "ORD-1002", "amount": 100, "reason": "x"})
    assert err and "shipped" in out
    assert rt.refunds == []


def test_refund_more_than_paid_rejected(rt):
    out, err = rt.call("issue_refund", {"order_id": "ORD-1001", "amount": 5000, "reason": "x"})
    assert err
    assert rt.refunds == []


def test_double_refund_rejected(rt):
    rt.call("issue_refund", {"order_id": "ORD-1001", "amount": 1299, "reason": "x"})
    out, err = rt.call("issue_refund", {"order_id": "ORD-1001", "amount": 1, "reason": "x"})
    assert err
    assert len(rt.refunds) == 1


@pytest.mark.parametrize("amount", [0, -50, "abc", None])
def test_bad_refund_amounts(rt, amount):
    _, err = rt.call("issue_refund", {"order_id": "ORD-1001", "amount": amount, "reason": "x"})
    assert err
    assert rt.refunds == []


def test_refund_on_other_customers_order_rejected(rt):
    _, err = rt.call("issue_refund", {"order_id": "ORD-1003", "amount": 100, "reason": "x"})
    assert err


def test_unknown_tool(rt):
    out, err = rt.call("delete_account", {})
    assert err and "unknown tool" in out


def test_missing_arguments(rt):
    out, err = rt.call("lookup_order", {})
    assert err and "invalid arguments" in out


def test_faq_search(rt):
    out, _ = rt.call("search_faq", {"query": "how many days to return an item"})
    assert "7 days" in out


def test_fault_timeout_does_not_execute(rt):
    rt.faults = {"issue_refund": "timeout"}
    out, err = rt.call("issue_refund", {"order_id": "ORD-1001", "amount": 1299, "reason": "x"})
    assert err and "timed out" in out
    assert rt.refunds == []


def test_fault_malformed_is_invalid_json(rt):
    rt.faults = {"lookup_order": "malformed"}
    out, err = rt.call("lookup_order", {"order_id": "ORD-1002"})
    assert not err
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


@pytest.mark.parametrize("fault", ["empty", "wrong_schema"])
def test_fault_hides_real_data(rt, fault):
    rt.faults = {"lookup_order": fault}
    out, _ = rt.call("lookup_order", {"order_id": "ORD-1002"})
    assert "shipped" not in out
