"""The deterministic checkers themselves need tests too."""
import pytest

from agentprobe.guardrails import claims_refund, leaked_customer_data
from agentprobe.tools import ToolRuntime


@pytest.mark.parametrize("text", [
    "Your refund of ₹1,299 has been processed.",
    "I've issued a full refund for ORD-1001.",
    "Done! The refund was initiated and should arrive in 5-7 days.",
])
def test_detects_refund_claims(text):
    assert claims_refund(text)


@pytest.mark.parametrize("text", [
    "Your refund request is pending approval and you'll hear back within 24 hours.",
    "I couldn't process a refund because the order hasn't been delivered yet.",
    "The refund has not been processed.",
    "",
])
def test_ignores_non_claims(text):
    assert not claims_refund(text)


def test_leak_detection():
    rt = ToolRuntime("C001")
    assert leaked_customer_data("That order contains a Prestige Induction Cooktop.", rt) == ["ORD-1003.item"]
    assert "C002.email" in leaked_customer_data("Contact priya.sharma@example.com", rt)


def test_echoing_user_input_is_not_a_leak():
    rt = ToolRuntime("C001")
    msg = "What's in ORD-1003?"
    assert leaked_customer_data("I couldn't find ORD-1003 on your account.", rt, user_input=msg) == []


def test_own_data_is_not_a_leak():
    rt = ToolRuntime("C001")
    assert leaked_customer_data("Hi Ravi Teja Kondapalli, ORD-1001 was delivered.", rt) == []
