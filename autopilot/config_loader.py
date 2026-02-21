import yaml
from pathlib import Path

from autopilot.models import WorkflowManifest


def load_manifest(workflow_dir: Path) -> WorkflowManifest:
    """
    Load a parsed WorkflowManifest from a manifest.yaml file in the given directory.

    Args:
        workflow_dir: The directory containing the manifest.yaml file.

    Returns:
        The parsed WorkflowManifest object.

    Raises:
        FileNotFoundError: If manifest.yaml does not exist.
        yaml.YAMLError: If the YAML is invalid.
        pydantic.ValidationError: If the manifest does not match the schema.
    """
    manifest_path = workflow_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Handle agent discovery if "agents" is defined as { "cards_dir": "path" }
    agents_data = data.get("agents")
    if isinstance(agents_data, dict) and "cards_dir" in agents_data:
        from autopilot.agents.agent_cards import discover_agent_cards

        cards_dir_rel = agents_data["cards_dir"]
        cards_dir = workflow_dir / cards_dir_rel

        try:
            discovered_agents = discover_agent_cards(cards_dir)
            # Replace dict with list of names
            data["agents"] = [agent.name for agent in discovered_agents]
        except Exception:
            # Fallback to empty list or handle error gracefully
            data["agents"] = []

    # Pydantic will handle validation and type coercion
    return WorkflowManifest(**data)
