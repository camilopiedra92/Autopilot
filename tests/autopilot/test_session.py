import pytest
from unittest.mock import AsyncMock, patch
from autopilot.core.session import RedisSessionService

@pytest.fixture
def mock_redis_client():
    client = AsyncMock()
    # Ensure keys() returns an empty list by default
    client.keys.return_value = []
    # Ensure mget returns empty list if no keys
    client.mget.return_value = []
    return client

@pytest.fixture
def session(mock_redis_client):
    with patch("autopilot.core.session.redis.from_url", return_value=mock_redis_client):
        return RedisSessionService(
            redis_url="redis://localhost:6379",
            session_id="test_sess_123"
        )

@pytest.mark.asyncio
async def test_redis_session_get_missing(session, mock_redis_client):
    mock_redis_client.get.return_value = None
    
    val = await session.get("my_key", default="fallback")
    assert val == "fallback"
    mock_redis_client.get.assert_called_once_with("autopilot:session:test_sess_123:my_key")

@pytest.mark.asyncio
async def test_redis_session_set_and_get_dict(session, mock_redis_client):
    # Setup the set behavior
    await session.set("user", {"name": "Alice"})
    mock_redis_client.set.assert_called_once_with(
        "autopilot:session:test_sess_123:user",
        '{"name": "Alice"}'
    )
    
    # Setup the get behavior
    mock_redis_client.get.return_value = '{"name": "Alice"}'
    val = await session.get("user")
    assert val == {"name": "Alice"}

@pytest.mark.asyncio
async def test_redis_session_set_and_get_string(session, mock_redis_client):
    # If the user sets a string, we JSON-encode it anyway to ensure safety based on our code logic
    # Actually, the code says:
    # if not isinstance(value, str): value = json.dumps(value)
    # So if it IS a string, it will be stored as string, but retrieved and tried as JSON.
    await session.set("token", "abc123yz")
    mock_redis_client.set.assert_called_once_with("autopilot:session:test_sess_123:token", "abc123yz")
    
    # Setup get
    mock_redis_client.get.return_value = "abc123yz"
    val = await session.get("token")
    
    # JSON decoding "abc123yz" will fail, and it will return the raw string
    assert val == "abc123yz"

@pytest.mark.asyncio
async def test_redis_session_delete(session, mock_redis_client):
    mock_redis_client.delete.return_value = 1
    deleted = await session.delete("old_key")
    assert deleted is True
    mock_redis_client.delete.assert_called_once_with("autopilot:session:test_sess_123:old_key")

@pytest.mark.asyncio
async def test_redis_session_clear(session, mock_redis_client):
    mock_redis_client.keys.return_value = ["autopilot:session:test_sess_123:k1", "autopilot:session:test_sess_123:k2"]
    
    await session.clear()
    
    mock_redis_client.keys.assert_called_once_with("autopilot:session:test_sess_123:*")
    mock_redis_client.delete.assert_called_once_with(
        "autopilot:session:test_sess_123:k1",
        "autopilot:session:test_sess_123:k2"
    )

@pytest.mark.asyncio
async def test_redis_session_snapshot(session, mock_redis_client):
    mock_redis_client.keys.return_value = [
        "autopilot:session:test_sess_123:user", 
        "autopilot:session:test_sess_123:count"
    ]
    mock_redis_client.mget.return_value = [
        '{"name": "Alice"}',
        "42"
    ]
    
    snapshot = await session.snapshot()
    
    assert snapshot == {
        "user": {"name": "Alice"},
        "count": 42
    }
