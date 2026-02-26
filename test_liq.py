import asyncio
import httpx
import time


async def test():
    now_ms = int(time.time() * 1000)
    window_ms = 3 * 60 * 1000  # last 3 minutes

    try:
        async with httpx.AsyncClient() as http:
            resp = await http.get(
                "https://fapi.binance.com/fapi/v1/allForceOrders",
                params={
                    "symbol": "BTCUSDT",
                    "startTime": now_ms - window_ms,
                    "limit": 100,
                },
                timeout=5.0,
            )
            resp.raise_for_status()
            orders = resp.json()
            print(f"Status Code: {resp.status_code}")
            print(f"Number of orders found: {len(orders)}")
            if len(orders) > 0:
                print(orders[0])
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test())
