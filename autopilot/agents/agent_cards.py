"""
Agent Card Loader — Discovers and parses .agent.yaml files.

Provides:
  - load_agent_card(yaml_path) → AgentCard
  - discover_agent_cards(agents_dir) → list[AgentCard]  (sorted by stage)

Each .agent.yaml file co-locates with its Python implementation,
describing WHAT the agent does (metadata, I/O, tools, guardrails)
while the .py defines HOW it works.
"""

from __future__ import annotations

from pathlib import Path

import structlog
import yaml

from autopilot.models import AgentCard

logger = structlog.get_logger(__name__)


def load_agent_card(yaml_path: str | Path) -> AgentCard:
    """
    Load and validate a single .agent.yaml file into an AgentCard.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        ValueError: If the YAML content fails Pydantic validation.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Agent card not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Agent card must be a YAML mapping, got {type(raw).__name__}: {path}")

    try:
        card = AgentCard(**raw)
    except Exception as e:
        raise ValueError(f"Invalid agent card {path.name}: {e}") from e

    logger.debug("agent_card_loaded", name=card.name, stage=card.stage, path=str(path))
    return card


def discover_agent_cards(agents_dir: str | Path) -> list[AgentCard]:
    """
    Discover all .agent.yaml files in a directory and return sorted AgentCards.

    Cards are sorted by `stage` (pipeline execution order).

    Args:
        agents_dir: Path to the agents directory (e.g., workflows/bank_to_ynab/agents/).

    Returns:
        List of AgentCard instances sorted by stage.
    """
    directory = Path(agents_dir)
    if not directory.is_dir():
        logger.warning("agent_cards_dir_not_found", path=str(directory))
        return []

    cards: list[AgentCard] = []
    yaml_files = sorted(directory.glob("*.agent.yaml"))

    for yaml_path in yaml_files:
        try:
            card = load_agent_card(yaml_path)
            cards.append(card)
        except (ValueError, FileNotFoundError) as e:
            logger.error("agent_card_load_failed", path=str(yaml_path), error=str(e))

    # Sort by pipeline stage
    cards.sort(key=lambda c: c.stage)

    logger.info(
        "agent_cards_discovered",
        directory=str(directory),
        count=len(cards),
        agents=[c.name for c in cards],
    )

    return cards
