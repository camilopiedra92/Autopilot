import asyncio
import httpx


async def test():
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.get(
                "https://fapi.binance.com/fapi/v1/ticker/24hr",
                params={"symbol": "BTCUSDT"},
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
            print("24h Ticker:", data)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test())
