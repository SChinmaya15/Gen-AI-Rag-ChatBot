import os
import sys
import traceback
import requests
from aiohttp import web
from botbuilder.core import (
    BotFrameworkAdapterSettings,
    BotFrameworkAdapter,
    TurnContext,
    ActivityHandler,
)
from botbuilder.schema import Activity
from dotenv import load_dotenv

from utils.db import init_db, save_conversation, get_tickets
# from utils.llm import extract_intent


# --- Init & config ---
load_dotenv()
init_db()

PORT = int(os.getenv("BOT_PORT", 3978))
APP_ID = os.getenv("MicrosoftAppId", "")
APP_PASSWORD = os.getenv("MicrosoftAppPassword", "")
MCP_URL = os.getenv("MCP_URL", "http://localhost:5000")   # mcp_server.py

settings = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD)
adapter = BotFrameworkAdapter(settings)

async def on_error(context: TurnContext, error: Exception):
    print(f"\n[on_turn_error] {error}", file=sys.stderr)
    traceback.print_exc()
    await context.send_activity("The bot hit an error. Please check server logs.")
adapter.on_turn_error = on_error

def call_mcp(tool: str, payload: dict):
    try:
        resp = requests.post(f"{MCP_URL}/{tool}", json=payload, timeout=15)
        if resp.status_code == 200:
            return {"ok": True, "data": resp.json()}
        return {"ok": False, "error": f"MCP HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"ok": False, "error": f"MCP exception: {e}"}

# Very simple in-memory conversation state per user
conversation_state = {}

class TeamsBot(ActivityHandler):
    async def on_message_activity(self, turn_context: TurnContext):
        user_id = turn_context.activity.from_property.id
        user_text = (turn_context.activity.text or "").strip()
        user_text_l = user_text.lower()

        if user_id not in conversation_state:
            conversation_state[user_id] = {"step": "greet"}

        state = conversation_state[user_id]
        reply = ""

        # quick commands
        if user_text_l == "my tickets":
            rows = get_tickets(user_id)
            if rows:
                lines = [f"{t[5]} | Ticket: {t[3]} | Software: {t[2]} | Exec: {t[4]}" for t in rows[:5]]
                reply = "Recent tickets:\n" + "\n".join(lines)
            else:
                reply = "No tickets found."
            await turn_context.send_activity(reply)
            save_conversation(user_id, user_text, reply)
            return

        # interactive flow
        if state["step"] == "greet":
            reply = "Hello! What software do you want to install?"
            state["step"] = "awaiting_software"

        elif state["step"] == "awaiting_software":
            state["software"] = user_text_l
            reply = f"Got it, you want **{state['software']}**. Should I create a ticket? (yes/no)"
            state["step"] = "confirm"

        elif state["step"] == "confirm":
            if user_text_l.startswith("y"):
                software = state.get("software", "unknown")
                await turn_context.send_activity(f"Creating ticket for **{software}**…")
                save_conversation(user_id, user_text, f"Creating ticket for {software}…")

                # Call MCP (ServiceNow + Rundeck) and include user_id
                res = call_mcp("trigger_installation", {"software": software, "user_id": user_id})

                if not res.get("ok"):
                    reply = f"❌ Failed: {res.get('error')}"
                else:
                    data = res["data"]
                    ticket = data.get("ticket", "UNKNOWN")
                    exec_id = data.get("execution_id", "N/A")
                    msg = data.get("message") or f"✅ Ticket `{ticket}` created. Execution ID: {exec_id}"
                    reply = msg

                state["step"] = "done"
            else:
                reply = "❌ Cancelled. Say anything to start again."
                state["step"] = "greet"

        else:
            reply = "Say 'hi' to start or type `my tickets` to view recent tickets."
            state["step"] = "greet"

        await turn_context.send_activity(reply)
        save_conversation(user_id, user_text, reply)

BOT = TeamsBot()

async def messages(req: web.Request) -> web.Response:
    if "application/json" not in req.headers.get("Content-Type", ""):
        return web.Response(status=415)

    body = await req.json()
    activity = Activity().deserialize(body)
    auth_header = req.headers.get("Authorization", "")

    response = await adapter.process_activity(activity, auth_header, BOT.on_turn)
    if response:
        return web.json_response(data=response.body, status=response.status)
    return web.Response(status=201)

APP = web.Application()
APP.router.add_post("/api/messages", messages)

if __name__ == "__main__":
    web.run_app(APP, host="localhost", port=PORT)
