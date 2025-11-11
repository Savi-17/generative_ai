import os
import uuid
import asyncio
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.function_tool import FunctionTool
from google.adk.apps.app import App, ResumabilityConfig

# Load environment variables (ensure .env exists)
load_dotenv()

# Configure retry options
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# -------------------------------------------------
# 1️⃣ Define the shipping tool
# -------------------------------------------------
LARGE_ORDER_THRESHOLD = 5

def place_shipping_order(num_containers: int, destination: str, tool_context: ToolContext) -> dict:
    """Places a shipping order. Requires approval if ordering more than 5 containers."""
    if num_containers <= LARGE_ORDER_THRESHOLD:
        return {
            "status": "approved",
            "order_id": f"ORD-{num_containers}-AUTO",
            "num_containers": num_containers,
            "destination": destination,
            "message": f"Order auto-approved: {num_containers} containers to {destination}",
        }

    # Pause for approval
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"⚠️ Large order: {num_containers} containers to {destination}. Approve?",
            payload={"num_containers": num_containers, "destination": destination},
        )
        return {"status": "pending", "message": f"Order for {num_containers} containers requires approval"}

    # Resumed after approval
    if tool_context.tool_confirmation.confirmed:
        return {
            "status": "approved",
            "order_id": f"ORD-{num_containers}-HUMAN",
            "num_containers": num_containers,
            "destination": destination,
            "message": f"Order approved: {num_containers} containers to {destination}",
        }
    else:
        return {"status": "rejected", "message": f"Order rejected: {num_containers} containers to {destination}"}


# -------------------------------------------------
# 2️⃣ Create the Agent
# -------------------------------------------------
shipping_agent = LlmAgent(
    name="shipping_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are a shipping coordinator assistant.

    When users request to ship containers:
    1. Use place_shipping_order with the number of containers and destination.
    2. If the order is pending, inform that approval is required.
    3. After result, provide clear summary including:
       - Order status (approved/rejected)
       - Order ID (if available)
       - Number of containers and destination.
    """,
    tools=[FunctionTool(func=place_shipping_order)],
)

# -------------------------------------------------
# 3️⃣ Make it resumable (for ADK Web UI)
# -------------------------------------------------
shipping_app = App(
    name="shipping_coordinator",
    root_agent=shipping_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

# -------------------------------------------------
# 4️⃣ Optional test runner (only for manual testing)
# -------------------------------------------------
session_service = InMemorySessionService()
shipping_runner = Runner(app=shipping_app, session_service=session_service)

async def run_shipping_workflow(query: str, auto_approve: bool = True):
    print("\n" + "=" * 60)
    print(f"User > {query}\n")

    session_id = f"order_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(app_name="shipping_coordinator", user_id="test_user", session_id=session_id)

    query_content = types.Content(role="user", parts=[types.Part(text=query)])
    events = []

    async for event in shipping_runner.run_async(user_id="test_user", session_id=session_id, new_message=query_content):
        events.append(event)

    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}")

    print("=" * 60 + "\n")

# -------------------------------------------------
# 🚀 Run only if directly executed, not when imported by ADK Web UI
# -------------------------------------------------
if __name__ == "__main__":
    asyncio.run(run_shipping_workflow("Ship 10 containers to Rotterdam"))
