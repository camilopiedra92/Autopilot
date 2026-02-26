import asyncio
import websockets
import json


async def test_ws():
    uri = "wss://fstream.binance.com/ws/btcusdt@forceOrder"
    try:
        async with websockets.connect(uri) as ws:
            print(
                "Connected to Binance forceOrder stream. Waiting 10s for liquidations..."
            )

            async def receive():
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    print("Liquidation:", data)

            # Run for 10 seconds only
            await asyncio.wait_for(receive(), timeout=10.0)
    except asyncio.TimeoutError:
        print("Test finished (10s timeout), no liquidations captured.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_ws())
