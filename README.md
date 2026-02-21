# Autopilot

An autonomous coding assistant and agent orchestration platform building **World-Class, Edge-First, Agentic Systems**.

## Overview

Autopilot provides edge-first, reliable, and scalable automated workflows. It includes connectors for services like Gmail, Telegram, Todoist, and Airtable, managed via a sophisticated async pipeline and directed acyclic graph (DAG) scheduler.

## Core Philosophy: "The Edge"

- **Headless API Topology**: The system operates as a pure backend API (JSON/Events). There is NO internal frontend or dashboard. All interactions occur via strictly secured API endpoints (`X-API-Key`) or trusted Webhooks (Pub/Sub).
- **Edge-First**: Logic runs as close to the data/user as possible utilizing lightweight, efficient patterns without monolithic state.
- **Agentic**: The system is composed of autonomous, intelligent agents that interact via standard protocols.
- **Google ADK Alignment**: Strictly follows the Google Agent Development Kit (ADK) patterns.

## Architecture Highlights

### Advanced Orchestration

Autopilot supports multi-strategy orchestration:

- **Sequential (`Pipeline`)**: Linear execution.
- **DAG (`DAGRunner`)**: Topological graph with parallel layers.
- **ReAct (`ReactRunner`)**: Reasoning + Acting loops with dynamic tool use.
- **Router (`RouterRunner`)**: LLM-based routing to best-fit sub-workflows.

Workflows can be defined cleanly via a Python API or entirely through declarative YAML definitions.

### Agent Factory & Guardrails

Agents are primarily created using standardized factory patterns (`create_platform_agent`), ensuring consistent observability, error handling, and model fallback support. Robust guardrails are chained via before/after callbacks to validate inputs and outputs (e.g., prompt injection prevention, data formatting constraints).

### Stateful Edge Execution

Engineered for scale-to-zero environments (like Google Cloud Run), Autopilot uses isolated session state (`BaseSessionService`: Redis or InMemory) and long-term semantic memory (`BaseMemoryService`: Chroma or InMemory) to safely manage context across ephemeral container executions. Keep-alive patterns are implemented for long-lived subscriptions.

### Tool & MCP Ecosystem

- **ToolRegistry**: A centralized repository supporting lazy connector resolution and native Google ADK `FunctionTool` generation.
- **MCP Native**: Out-of-the-box support for the Model Context Protocol (MCP) via `MCPBridge` to seamlessly plug in external context servers.
- **Callbacks**: Advanced auth tracking, rate-limiting, and audit logging built into global before/after tool hooks.

### Agent Bus (A2A Communication)

A typed, async pub/sub messaging bus (`AgentBus`) facilitates decoupled inter-agent communication, allowing distinct components to emit and respond to events securely at runtime.
