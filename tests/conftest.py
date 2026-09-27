import os

import pytest

from agentprobe.agent import SupportAgent
from agentprobe.tools import ToolRuntime
from agentprobe.tracer import Tracer


def pytest_collection_modifyitems(config, items):
    if os.getenv("ANTHROPIC_API_KEY"):
        return
    skip = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def tracer():
    return Tracer()


@pytest.fixture
def runtime():
    return ToolRuntime(customer_id="C001")


@pytest.fixture
def make_agent(runtime, tracer):
    """Build an agent. Defaults to the shared runtime and prompt from AGENTPROBE_PROMPT (v2)."""
    def _make(rt=None, prompt_version=None):
        return SupportAgent(
            rt or runtime,
            prompt_version=prompt_version or os.getenv("AGENTPROBE_PROMPT", "v2"),
            tracer=tracer,
        )
    return _make
