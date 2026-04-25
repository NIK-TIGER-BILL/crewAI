"""Test A2A skill ID resolution in delegation context."""

from unittest.mock import MagicMock

import pytest

from crewai.a2a.config import A2AClientConfig, A2AConfig


def _create_mock_agent_card_dict(endpoint: str = "http://test-endpoint.com/", skill_id: str = "Research") -> dict:
    """Create a mock agent card as a dict with skills."""
    return {
        "name": "Test Agent",
        "url": endpoint,
        "skills": [
            {
                "id": skill_id,
                "name": skill_id,
                "description": "Test skill",
            }
        ],
    }


def _create_mock_agent_card_object(endpoint: str = "http://test-endpoint.com/", skill_id: str = "Research"):
    """Create a mock agent card object with skills."""
    mock_skill = MagicMock()
    mock_skill.id = skill_id

    mock_card = MagicMock()
    mock_card.name = "Test Agent"
    mock_card.url = endpoint
    mock_card.skills = [mock_skill]
    return mock_card


class TestResolveAgentId:
    """Tests for _resolve_agent_id function."""

    def test_direct_endpoint_match(self):
        """Should return the agent_id when it matches an endpoint directly."""
        from crewai.a2a.wrapper import _resolve_agent_id

        agents = [
            A2AConfig(endpoint="http://agent1.com"),
            A2AConfig(endpoint="http://agent2.com"),
        ]

        result = _resolve_agent_id("http://agent2.com", agents, None)
        assert result == "http://agent2.com"

    def test_resolve_skill_id_dict_card(self):
        """Should resolve skill ID to endpoint when agent_cards is a dict."""
        from crewai.a2a.wrapper import _resolve_agent_id

        agents = [A2AConfig(endpoint="http://test-endpoint.com/")]
        agent_cards = {"http://test-endpoint.com/": _create_mock_agent_card_dict("http://test-endpoint.com/", "Research")}

        result = _resolve_agent_id("Research", agents, agent_cards)
        assert result == "http://test-endpoint.com/"

    def test_resolve_skill_id_object_card(self):
        """Should resolve skill ID to endpoint when agent_cards is an object."""
        from crewai.a2a.wrapper import _resolve_agent_id

        agents = [A2AConfig(endpoint="http://test-endpoint.com/")]
        agent_cards = {"http://test-endpoint.com/": _create_mock_agent_card_object("http://test-endpoint.com/", "Research")}

        result = _resolve_agent_id("Research", agents, agent_cards)
        assert result == "http://test-endpoint.com/"

    def test_no_match_returns_original(self):
        """Should return original agent_id when no match is found."""
        from crewai.a2a.wrapper import _resolve_agent_id

        agents = [A2AConfig(endpoint="http://test-endpoint.com/")]
        agent_cards = {"http://test-endpoint.com/": _create_mock_agent_card_dict("http://test-endpoint.com/", "Research")}

        result = _resolve_agent_id("UnknownSkill", agents, agent_cards)
        assert result == "UnknownSkill"

    def test_no_agent_cards_returns_original(self):
        """Should return original agent_id when agent_cards is None."""
        from crewai.a2a.wrapper import _resolve_agent_id

        agents = [A2AConfig(endpoint="http://test-endpoint.com/")]

        result = _resolve_agent_id("Research", agents, None)
        assert result == "Research"

    def test_multiple_agents_resolves_correct_one(self):
        """Should resolve skill ID to the correct endpoint among multiple agents."""
        from crewai.a2a.wrapper import _resolve_agent_id

        agents = [
            A2AConfig(endpoint="http://agent1.com"),
            A2AConfig(endpoint="http://agent2.com"),
        ]
        agent_cards = {
            "http://agent1.com": _create_mock_agent_card_dict("http://agent1.com", "Analysis"),
            "http://agent2.com": _create_mock_agent_card_dict("http://agent2.com", "Research"),
        }

        result = _resolve_agent_id("Research", agents, agent_cards)
        assert result == "http://agent2.com"


class TestPrepareDelegationContext:
    """Tests for _prepare_delegation_context with skill IDs."""

    def test_prepare_delegation_context_with_skill_id(self):
        """Should resolve skill ID in agent_response to endpoint URL."""
        from crewai.a2a.wrapper import _prepare_delegation_context
        from crewai.a2a.types import AgentResponseProtocol
        from crewai import Agent, Task

        a2a_config = A2AConfig(endpoint="http://test-endpoint.com/")
        agent = Agent(
            role="test manager",
            goal="coordinate",
            backstory="test",
            a2a=a2a_config,
        )
        task = Task(description="test", expected_output="test", agent=agent)

        class MockResponse:
            is_a2a = True
            message = "Please help"
            a2a_ids = ("Research",)

        agent_cards = {
            "http://test-endpoint.com/": _create_mock_agent_card_dict("http://test-endpoint.com/", "Research")
        }

        ctx = _prepare_delegation_context(agent, MockResponse(), task, "test", agent_cards)

        assert ctx.agent_id == "http://test-endpoint.com/"
        assert ctx.agent_config.endpoint == "http://test-endpoint.com/"
