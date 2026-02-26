import httpx


async def test():
    # Attempting to fetch Coinglass liquidation heat map data (usually public)
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.get(
                "https://open-api.coinglass.com/public/v2/liquidation_history",
                params={"symbol": "BTC", "time_type": "all"},
                timeout=5.0,
                headers={"accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            print("Coinglass:", data)
    except Exception as e:
        print(f"Error Coinglass: {e}")


test()
