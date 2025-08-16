import os
import pandas as pd

INCIDENT_FILE = "data/incident_log.xlsx"

def write_new_incident(req):
    row = {
        "RequestID": req.request_id,
        "User": req.user_name,
        "Application": req.application,
        "Version": req.version,
        "Remarks": req.remarks,
        "Status": req.status
    }
    if os.path.exists(INCIDENT_FILE):
        df = pd.read_excel(INCIDENT_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_excel(INCIDENT_FILE, index=False)
