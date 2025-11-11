import os
import uuid
from dotenv import load_dotenv
from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.adk.tools.function_tool import FunctionTool

# ----------------------------------------------------------
# 0️⃣ Load environment and setup
# ----------------------------------------------------------
load_dotenv()
print("✅ Environment loaded.")

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# ----------------------------------------------------------
# 1️⃣ Connect to MCP Server
# ----------------------------------------------------------
mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
            tool_filter=["getTinyImage"],  # demo image gen tool
        ),
        timeout=30,
    )
)
print("✅ MCP Image server connected.")

# ----------------------------------------------------------
# 2️⃣ Define custom tool with approval
# ----------------------------------------------------------
BULK_THRESHOLD = 1

def request_image_generation(prompt: str, num_images: int, tool_context: ToolContext) -> dict:
    """Handles approval logic for image generation."""
    print(f"🔧 Running request_image_generation: prompt='{prompt}', num_images={num_images}")

    # Case 1: Auto-approved single image
    if num_images <= BULK_THRESHOLD:
        return {
            "status": "approved",
            "message": f"✅ Auto-approved: {num_images} image(s) for '{prompt}'.",
        }

    # Case 2: Pause and ask for approval
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"⚠️ Bulk request for {num_images} images. Approve generation for '{prompt}'?",
            payload={"prompt": prompt, "num_images": num_images},
        )
        return {
            "status": "pending",
            "message": f"⏸️ Awaiting approval for {num_images} image(s).",
        }

    # Case 3: Resuming after approval
    if tool_context.tool_confirmation.confirmed:
        return {
            "status": "approved",
            "message": f"✅ Approved bulk generation: {num_images} image(s) for '{prompt}'.",
        }
    else:
        return {
            "status": "rejected",
            "message": f"❌ Rejected bulk generation for '{prompt}'.",
        }

# ----------------------------------------------------------
# 3️⃣ Build the LLM Agent
# ----------------------------------------------------------
image_agent = LlmAgent(
    name="image_generation_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are an AI assistant that generates images.
    Use request_image_generation() to confirm user requests.
    If approved, use the MCP getTinyImage tool to produce a small demo image.
    Always respond with a summary of what was done or why it was not done.
    """,
    tools=[FunctionTool(func=request_image_generation), mcp_image_server],
)
print("✅ LLM Agent created.")

# ----------------------------------------------------------
# 4️⃣ Wrap as resumable app for ADK Web
# ----------------------------------------------------------
image_app = App(
    name="image_generation_agent",
    model="gemini-2.0-flash",
    description="Converts currencies and applies transaction fees.",
    instruction="You convert currencies using get_fee_for_payment_method() and get_exchange_rate() tools.",
    tools=[get_fee_for_payment_method, get_exchange_rate],
    root_agent=image_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)


session_service = InMemorySessionService()
image_runner = Runner(app=image_app, session_service=session_service)
print("✅ Runner ready.")

# ----------------------------------------------------------
# 5️⃣ Define helper test workflow (for debugging)
# ----------------------------------------------------------
import asyncio

async def run_image_workflow(prompt: str, num_images: int):
    print("\n" + "="*60)
    print(f"🧠 Prompt: {prompt} | Count: {num_images}")
    session_id = f"img_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name="image_generation_agent", user_id="test_user", session_id=session_id
    )
    query_content = types.Content(role="user", parts=[types.Part(text=f"Generate {num_images} images for {prompt}")])
    async for event in image_runner.run_async(user_id="test_user", session_id=session_id, new_message=query_content):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}")
    print("="*60 + "\n")

# ----------------------------------------------------------
# 6️⃣ Export root agent for ADK Web
# ----------------------------------------------------------
root_agent = image_app
print("✅ root_agent exported for ADK Web.")
