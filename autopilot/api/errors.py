from fastapi import Request
from fastapi.responses import JSONResponse
import structlog

from autopilot.errors import AutoPilotError

logger = structlog.get_logger(__name__)

async def autopilot_error_handler(request: Request, exc: AutoPilotError) -> JSONResponse:
    """
    Global exception handler for the AutoPilot platform.
    Converts internal, structured exceptions into standard JSON API responses.
    """
    error_data = exc.to_dict()
    
    # We log it here so every handled API error is visible in structured logs.
    logger.warning(
        "api_error_handled",
        path=request.url.path,
        error_code=error_data.get("error_code"),
        message=error_data.get("message"),
        status_code=exc.http_status,
        detail=error_data.get("detail")
    )

    return JSONResponse(
        status_code=exc.http_status,
        content={"error": error_data}
    )
