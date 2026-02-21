import os
from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED

# Expected header: X-API-Key
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Validates the X-API-Key header against the API_KEY_SECRET environment variable.
    Required for internal management routes.
    """
    secret = os.getenv("API_KEY_SECRET")
    
    # If the server runs without a secret configured, reject all requests securely.
    if not secret:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="API_KEY_SECRET is not configured on the server",
        )
        
    if api_key != secret:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    
    return api_key
