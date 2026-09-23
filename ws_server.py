import asyncio
import json
import os
import websockets

connected_clients = set()
ai_twin = None  # set once the heavy models finish loading, in the background

async def telemetry_listener():
    uri = os.environ.get(
        "VIRTUAL_ENGINE_WS_URL",
        "ws://localhost:8080/telemetry"
    )
    while True:
        try:
            async with websockets.connect(uri) as ws:
                print("[AI Engine] Connected to Virtual Engine telemetry stream on port 8080.")
                async for message in ws:
                    if ai_twin is None:
                        # Models still loading in the background — skip this packet.
                        continue

                    packet = json.loads(message)
                    result = ai_twin.process(packet)
                    decision = {
                        "timestamp": result["timestamp"],
                        "current_state": result["current_state"],
                        "ai": result["ai_prediction"],
                        "health_score": result["health_score"],
                    }

                    if connected_clients and decision:
                        payload = json.dumps(decision)
                        await asyncio.gather(*[client.send(payload) for client in connected_clients])
        except Exception:
            await asyncio.sleep(2)


async def decision_server(websocket):
    connected_clients.add(websocket)
    print("[AI Engine] Dashboard client connected to /decision")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)


async def load_models():
    """Loads the ML models in a background thread so it never blocks
    the event loop or delays opening the port."""
    global ai_twin
    from ai_digital_twin import AIDigitalTwin  # deferred import — this is what's actually slow
    loop = asyncio.get_event_loop()
    ai_twin = await loop.run_in_executor(None, AIDigitalTwin)
    print("[AI Engine] Models loaded — predictions active.")


async def main():
    port = int(os.environ.get("PORT", 8081))

    # Open the port FIRST — this is what Render's health check needs to see immediately.
    server = await websockets.serve(decision_server, "0.0.0.0", port)
    print(f"AI/Math Decision WebSocket active on ws://0.0.0.0:{port}/decision")

    await asyncio.gather(
        server.wait_closed(),
        telemetry_listener(),
        load_models(),
    )

if __name__ == "__main__":
    asyncio.run(main())