import os
import streamlit as st
from db import SessionLocal, create_request, get_all_requests, get_request_by_id, update_request_status
from excel_utils import write_new_incident
from rundeck_stub import simulate_installation
from llm_utils import parse_request_text

st.set_page_config(page_title="Teams Incident Bot Prototype", layout="wide")

st.title("💬 Teams Bot Prototype with LLM")

st.sidebar.header("Request List")

with SessionLocal() as session:
    requests = get_all_requests(session)
    req_options = {f"{r.request_id} | {r.application} | {r.status}": r.request_id for r in requests}
selected = st.sidebar.selectbox("Select a request", list(req_options.keys()) if req_options else ["None"])
if selected != "None":
    with SessionLocal() as session:
        r = get_request_by_id(session, req_options[selected])
        if r:
            st.write(f"**Request ID**: {r.request_id}")
            st.write(f"**Application**: {r.application}")
            st.write(f"**Version**: {r.version or '-'}")
            st.write(f"**Remarks**: {r.remarks or '-'}")
            st.write(f"**Status**: {r.status}")
            col1, col2, col3 = st.columns(3)
            if col1.button("Approve"):
                update_request_status(session, r, "Approved", approver="Supervisor")
                st.success("Request Approved")
            if col2.button("Reject"):
                update_request_status(session, r, "Rejected", approver="Supervisor")
                st.error("Request Rejected")
            if col3.button("Trigger Install"):
                update_request_status(session, r, "In Progress")
                simulate_installation(r)
                update_request_status(session, r, "Completed")
                st.info("Installation completed!")

st.subheader("Create Request via Form")
with st.form("req_form"):
    user_name = st.text_input("User Name")
    app_name = st.text_input("Application Name")
    version = st.text_input("Version (optional)")
    remarks = st.text_area("Remarks")
    submitted = st.form_submit_button("Submit Request")
    if submitted:
        with SessionLocal() as session:
            req = create_request(session, user_name, app_name, version, remarks)
            write_new_incident(req)
        st.success(f"Request {req.request_id} submitted.")

st.subheader("Chat with Bot (LLM-powered)")
user_chat = st.text_input("Type your request")
if user_chat:
    parsed = parse_request_text(user_chat)
    with SessionLocal() as session:
        req = create_request(session, user_name="ChatUser", application=parsed.get("application"), version=parsed.get("version"), remarks=parsed.get("remarks"))
        write_new_incident(req)
    st.success(f"✅ Got it! Created request {req.request_id} for {parsed.get('application')} {parsed.get('version') or ''}")
