import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from utils.db import init_db, save_ticket

load_dotenv()
app = Flask(__name__)
init_db()

# --- ServiceNow config ---
SN_URL  = os.getenv("SERVICENOW_URL")  # e.g., https://dev350410.service-now.com
SN_USER = os.getenv("SERVICENOW_USER") # e.g., admin
SN_PASS = os.getenv("SERVICENOW_PASS")

# --- Rundeck config ---
RUNDECK_URL      = os.getenv("RUNDECK_URL", "http://localhost:4440")
RUNDECK_JOB_ID   = os.getenv("RUNDECK_JOB_ID")  # UUID
RUNDECK_API_TOKEN = os.getenv("RUNDECK_API_TOKEN")

def create_ticket_in_servicenow(software: str) -> dict:
    """
    Returns: {"ok": True, "ticket": "INC0012345"} on success,
             {"ok": False, "error": "..."} on failure.
    """
    try:
        url = f"{SN_URL}/api/now/table/incident"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        payload = {"short_description": f"Install request for {software}"}

        resp = requests.post(
            url,
            auth=(SN_USER, SN_PASS),
            headers=headers,
            json=payload,
            timeout=15
        )

        print("SNOW STATUS:", resp.status_code)
        print("SNOW BODY:", resp.text)

        if resp.status_code in (200, 201):
            data = resp.json()
            number = data.get("result", {}).get("number")
            if number:
                return {"ok": True, "ticket": number}
            return {"ok": False, "error": "Missing ticket number in response."}
        else:
            return {"ok": False, "error": f"ServiceNow HTTP {resp.status_code}: {resp.text}"}

    except Exception as e:
        return {"ok": False, "error": f"ServiceNow exception: {e}"}

def trigger_rundeck_job(software: str) -> dict:
    """
    Returns: {"ok": True, "execution_id": "123"} on success,
             {"ok": False, "error": "..."} on failure.
    """
    try:
        url = f"{RUNDECK_URL}/api/45/job/{RUNDECK_JOB_ID}/run"
        headers = {
            "X-Rundeck-Auth-Token": RUNDECK_API_TOKEN,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        # Optional: pass an arg to job
        payload = {"argString": f"-software {software}"}

        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        print("RUNDECK STATUS:", resp.status_code, resp.text)

        if resp.status_code == 200:
            data = resp.json()
            exec_id = data.get("id")
            if exec_id:
                return {"ok": True, "execution_id": str(exec_id)}
            return {"ok": False, "error": "Missing execution id in response."}
        else:
            return {"ok": False, "error": f"Rundeck HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"ok": False, "error": f"Rundeck exception: {e}"}

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/trigger_installation", methods=["POST"])
def trigger_installation():
    data = request.get_json(force=True) or {}
    software = data.get("software", "").strip()
    user_id = data.get("user_id", "local")

    if not software:
        return jsonify({"error": "Software name missing"}), 400

    # 1) Create ServiceNow ticket (strict success)
    sn = create_ticket_in_servicenow(software)
    if not sn.get("ok"):
        return jsonify({"error": f"ServiceNow ticket failed: {sn.get('error')}" }), 502

    ticket = sn["ticket"]

    # 2) Trigger Rundeck job
    rd = trigger_rundeck_job(software)
    if not rd.get("ok"):
        # Store ticket even if Rundeck failed (we still raised an incident)
        save_ticket(user_id, software, ticket, execution_id="FAILED")
        return jsonify({
            "ticket": ticket,
            "error": f"Rundeck failed: {rd.get('error')}"
        }), 502

    execution_id = rd["execution_id"]

    # 3) Persist to SQLite
    save_ticket(user_id, software, ticket, execution_id)

    return jsonify({
        "ticket": ticket,
        "execution_id": execution_id,
        "message": f"✅ Ticket {ticket} created and installation of {software} started"
    })

if __name__ == "__main__":
    # Runs on http://localhost:5000
    app.run(host="127.0.0.1", port=5000, debug=True)
