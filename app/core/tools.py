# import os
# import httpx
# from datetime import datetime
# from langchain_core.tools import tool

# CONVEX_SITE_URL = os.getenv("CONVEX_SITE_URL")


# @tool
# def create_calendar_event(
#     project_id: str,
#     title: str,
#     description: str,
#     event_type: str,  # "event" | "milestone"
#     start_iso: str,  # ISO 8601 e.g. "2025-04-22T00:00:00"
#     end_iso: str,
# ) -> str:
#     """
#     Create a calendar event for a project.
#     event_type must be one of: event, milestone.
#     start_iso and end_iso must be ISO 8601 strings.
#     All events are created as all-day events by default.
#     Never ask the user for project_id — it is injected automatically.
#     """
#     # ── This return value is NEVER reached in normal flow ──
#     # The graph routes this tool_call to hitl_calendar node before ToolNode runs.
#     # Kept as a descriptor so the LLM knows the tool schema.
#     return "intercepted"


# async def write_calendar_event_to_convex(payload: dict) -> str:
#     """Actual HTTP call to Convex — called by hitl_calendar node after approval."""
#     try:
#         async with httpx.AsyncClient() as client:
#             response = await client.post(
#                 f"{CONVEX_SITE_URL}/createCalendarEvent",
#                 json=payload,
#                 timeout=10,
#             )
#             response.raise_for_status()
#             result = response.json()
#             return (
#                 f"✅ Calendar event created: '{payload['title']}' "
#                 f"(id: {result.get('id', 'unknown')})"
#             )
#     except httpx.HTTPError as e:
#         return f"❌ Failed to create calendar event: {e}"
