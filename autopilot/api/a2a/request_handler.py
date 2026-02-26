"""
A2A Request Handler — Bridges A2A Protocol to Autopilot WorkflowRegistry.

Implements the A2A SDK's RequestHandler ABC, mapping incoming A2A messages
to workflow executions. Each message/send call extracts the target workflow
from the message payload, executes it via BaseWorkflow.run(), and returns
the result as an A2A Task.

Message protocol:
  The first TextPart of the A2A message MUST be valid JSON containing:
    {"workflow": "workflow_id", ...rest_is_trigger_data}
"""

import json
import structlog
from collections import OrderedDict
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import RequestHandler
from a2a.server.request_handlers.request_handler import ServerError
from a2a.types import (
    Artifact,
    Message,
    MessageSendParams,
    Part,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskQueryParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)

from autopilot.errors import A2ATaskNotFoundError, A2AWorkflowNotFoundError
from autopilot.models import RunStatus, TriggerType
from autopilot.registry import WorkflowRegistry

logger = structlog.get_logger(__name__)

# Maximum number of tasks to keep in the in-memory store
_MAX_TASK_STORE_SIZE = 1000

# Mapping from platform RunStatus to A2A TaskState
_STATUS_MAP: dict[RunStatus, TaskState] = {
    RunStatus.PENDING: TaskState.submitted,
    RunStatus.RUNNING: TaskState.working,
    RunStatus.SUCCESS: TaskState.completed,
    RunStatus.FAILED: TaskState.failed,
    RunStatus.SKIPPED: TaskState.completed,
}


def _map_status(run_status: RunStatus) -> TaskState:
    """Map platform RunStatus to A2A TaskState."""
    return _STATUS_MAP.get(run_status, TaskState.unknown)


def _extract_workflow_request(
    params: MessageSendParams,
) -> tuple[str, dict[str, Any]]:
    """Extract workflow ID and trigger data from an A2A message.

    The first TextPart of the message must be JSON with a "workflow" key.

    Args:
        params: The A2A MessageSendParams.

    Returns:
        Tuple of (workflow_id, trigger_data).

    Raises:
        A2AWorkflowNotFoundError: If no workflow key is found in the message.
    """
    for part in params.message.parts:
        if isinstance(part.root, TextPart):
            try:
                data = json.loads(part.root.text)
                if isinstance(data, dict) and "workflow" in data:
                    workflow_id = data.pop("workflow")
                    return workflow_id, data
            except (json.JSONDecodeError, TypeError):
                continue

    raise A2AWorkflowNotFoundError(
        "Message must contain a TextPart with JSON: "
        '{"workflow": "<workflow_id>", ...trigger_data}'
    )


def _now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _unsupported(message: str) -> ServerError:
    """Create a ServerError wrapping UnsupportedOperationError."""
    return ServerError(error=UnsupportedOperationError(message=message))


class AutopilotA2ARequestHandler(RequestHandler):
    """A2A RequestHandler that bridges to Autopilot's WorkflowRegistry.

    Handles message/send by extracting the workflow ID from the message,
    executing it via BaseWorkflow.run(), and returning the result as
    an A2A Task with artifacts.
    """

    def __init__(self, registry: WorkflowRegistry) -> None:
        self._registry = registry
        self._tasks: OrderedDict[str, Task] = OrderedDict()

    def _store_task(self, task: Task) -> None:
        """Store a task, evicting oldest if over capacity."""
        self._tasks[task.id] = task
        while len(self._tasks) > _MAX_TASK_STORE_SIZE:
            self._tasks.popitem(last=False)

    async def on_message_send(
        self,
        params: MessageSendParams,
        context: ServerCallContext | None = None,
    ) -> Task:
        """Handle message/send — execute a workflow and return the Task."""
        workflow_id, trigger_data = _extract_workflow_request(params)

        wf = self._registry.get(workflow_id)
        if not wf:
            raise A2AWorkflowNotFoundError(
                f"Workflow '{workflow_id}' not found in registry"
            )

        task_id = uuid4().hex
        context_id = uuid4().hex

        logger.info(
            "a2a_task_started",
            task_id=task_id,
            workflow=workflow_id,
        )

        # Execute the workflow
        run = await wf.run(TriggerType.MANUAL, trigger_data)

        # Build result artifacts
        artifacts = [
            Artifact(
                artifactId=uuid4().hex,
                name=f"{workflow_id}_result",
                parts=[Part(root=TextPart(text=json.dumps(run.result, default=str)))],
            )
        ]

        task = Task(
            id=task_id,
            contextId=context_id,
            status=TaskStatus(
                state=_map_status(run.status),
                timestamp=_now_iso(),
            ),
            artifacts=artifacts if run.result else None,
            history=[params.message],
            metadata={
                "workflow_id": workflow_id,
                "run_id": run.id,
                "duration_ms": run.duration_ms,
            },
        )

        self._store_task(task)

        logger.info(
            "a2a_task_completed",
            task_id=task_id,
            workflow=workflow_id,
            state=task.status.state.value,
            duration_ms=run.duration_ms,
        )

        return task

    async def on_message_send_stream(
        self,
        params: MessageSendParams,
        context: ServerCallContext | None = None,
    ) -> AsyncGenerator[
        Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
    ]:
        """Handle message/stream — stream task status updates during execution."""
        workflow_id, trigger_data = _extract_workflow_request(params)

        wf = self._registry.get(workflow_id)
        if not wf:
            raise A2AWorkflowNotFoundError(
                f"Workflow '{workflow_id}' not found in registry"
            )

        task_id = uuid4().hex
        context_id = uuid4().hex

        # Emit: submitted
        yield TaskStatusUpdateEvent(
            taskId=task_id,
            contextId=context_id,
            status=TaskStatus(
                state=TaskState.submitted,
                timestamp=_now_iso(),
            ),
            final=False,
        )

        # Emit: working
        yield TaskStatusUpdateEvent(
            taskId=task_id,
            contextId=context_id,
            status=TaskStatus(
                state=TaskState.working,
                timestamp=_now_iso(),
            ),
            final=False,
        )

        # Execute the workflow
        run = await wf.run(TriggerType.MANUAL, trigger_data)

        # Emit artifact if there are results
        if run.result:
            yield TaskArtifactUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                artifact=Artifact(
                    artifactId=uuid4().hex,
                    name=f"{workflow_id}_result",
                    parts=[
                        Part(root=TextPart(text=json.dumps(run.result, default=str)))
                    ],
                ),
            )

        # Emit: final status (completed/failed)
        final_state = _map_status(run.status)
        final_status = TaskStatus(
            state=final_state,
            timestamp=_now_iso(),
        )

        # Store the task
        task = Task(
            id=task_id,
            contextId=context_id,
            status=final_status,
            history=[params.message],
            metadata={
                "workflow_id": workflow_id,
                "run_id": run.id,
                "duration_ms": run.duration_ms,
            },
        )
        self._store_task(task)

        yield TaskStatusUpdateEvent(
            taskId=task_id,
            contextId=context_id,
            status=final_status,
            final=True,
        )

    async def on_get_task(
        self,
        params: TaskQueryParams,
        context: ServerCallContext | None = None,
    ) -> Task | None:
        """Handle tasks/get — retrieve a task by ID."""
        task = self._tasks.get(params.id)
        if not task:
            raise A2ATaskNotFoundError(f"Task '{params.id}' not found")
        return task

    async def on_cancel_task(
        self,
        params: TaskIdParams,
        context: ServerCallContext | None = None,
    ) -> Task | None:
        """Unsupported — Autopilot workflows are atomic, non-cancellable."""
        raise _unsupported("Task cancellation is not supported")

    async def on_set_task_push_notification_config(
        self,
        params,
        context: ServerCallContext | None = None,
    ):
        """Unsupported — push notifications are not supported."""
        raise _unsupported("Push notifications are not supported")

    async def on_get_task_push_notification_config(
        self,
        params,
        context: ServerCallContext | None = None,
    ):
        """Unsupported — push notifications are not supported."""
        raise _unsupported("Push notifications are not supported")

    async def on_delete_task_push_notification_config(
        self,
        params,
        context: ServerCallContext | None = None,
    ) -> None:
        """Unsupported — push notifications are not supported."""
        raise _unsupported("Push notifications are not supported")

    async def on_list_task_push_notification_config(
        self,
        params,
        context: ServerCallContext | None = None,
    ):
        """Unsupported — push notifications are not supported."""
        raise _unsupported("Push notifications are not supported")

    async def on_resubscribe_to_task(
        self,
        params: TaskIdParams,
        context: ServerCallContext | None = None,
    ):
        """Unsupported — task resubscription is not supported."""
        raise _unsupported("Task resubscription is not supported")
