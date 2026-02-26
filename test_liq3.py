import asyncio
import httpx


async def test():
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.get(
                "https://api.coinglass.com/public/v2/liquidation_history",
                params={"symbol": "BTC", "time_type": "all", "limit": 10},
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
            print("Coinglass:", data)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test())
