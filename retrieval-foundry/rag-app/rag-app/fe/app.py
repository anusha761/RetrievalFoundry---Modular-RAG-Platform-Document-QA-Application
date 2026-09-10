import os
import base64
from pathlib import Path
import requests
import streamlit as st

# Configure wide layout
st.set_page_config(layout="wide", page_title="RetrievalFoundry - Modular RAG Platform - Document QA App")

# ==============================================================================
# 🔐 1. BACKEND INITIALIZATION & CONFIGS
# ==============================================================================
BACKEND_URL = "http://localhost:8000"
API_KEY = "dev-api-key-12345"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
DOCS_DIR = Path(r"D:\retrieval-foundry\rag-engine\data\Files")


def _get_citation_section(section_path):
    if not section_path:
        return ""
    last = section_path[-1]
    if last.lower().startswith("table") and len(section_path) > 1:
        return section_path[-2]
    return last


def _format_citation_label(cit, number=None):
    section = _get_citation_section(cit.get("section_path", []))
    parts = [cit["file_name"]]
    if section:
        parts.append(section)
    parts.append(f"p.{cit['page_number']}")
    prefix = f"[{number}] " if number is not None else ""
    return f"{prefix}Source: {' - '.join(parts)}"

# Initialize state trackers
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

DOCUMENT_CATALOG = {
    pdf.name: {
        "file_id": pdf.stem,
        "path": str(pdf),
    }
    for pdf in sorted(DOCS_DIR.glob("*.pdf"))
}

# ==============================================================================
# 📁 2. LEFT SIDEBAR: DOCUMENT SELECTION WITH CHECKBOXES
# ==============================================================================
with st.sidebar:
    st.title("📚 Document Library")
    st.caption("Select documents to include in the analysis")
    st.markdown("---")

    selected_file_ids = []
    for doc_name, doc_info in DOCUMENT_CATALOG.items():
        col1, col2 = st.columns([0.75, 0.25])
        with col1:
            checked = st.checkbox(doc_name, key=f"chk_{doc_name}")
        with col2:
            if os.path.exists(doc_info["path"]):
                with open(doc_info["path"], "rb") as f:
                    pdf_bytes = f.read()
                    b64 = base64.b64encode(pdf_bytes).decode()
                    st.markdown(
                        f'<a href="data:application/pdf;base64,{b64}" target="_blank" style="text-decoration:none;font-size:13px;">View</a>',
                        unsafe_allow_html=True
                    )

        if checked:
            selected_file_ids.append(doc_info["file_id"])

    st.markdown("---")
    st.caption(f"**{len(selected_file_ids)}** document(s) selected")

    if len(selected_file_ids) == 0:
        st.warning("Select at least 1 document to start querying.")
    elif len(selected_file_ids) > 10:
        st.error("Maximum 10 documents allowed.")
        selected_file_ids = selected_file_ids[:10]

# ==============================================================================
# 💬 3. MAIN CHAT INTERFACE
# ==============================================================================
st.title("🤖 RetrievalFoundry - Modular RAG Platform - Document QA App")

if selected_file_ids:
    doc_names = [name for name, info in DOCUMENT_CATALOG.items() if info["file_id"] in selected_file_ids]
    chips_html = " ".join([f'<span style="background-color:#e0f2f1;border:1px solid #00897b;border-radius:16px;padding:4px 12px;margin:2px;display:inline-block;font-size:13px;">📄 {name}</span>' for name in doc_names])
    st.markdown(f'<div style="margin-bottom:10px;">{chips_html}</div>', unsafe_allow_html=True)
else:
    st.info("👈 Select documents from the sidebar to start asking questions.")

# 🚀 QUICK FIX ADDED HERE: Wrap the chat display inside a scrollable container with a fixed height.
# This gives you native chat auto-scrolling behavior immediately!
chat_container = st.container(height=600)

with chat_container:
    # Render chat history inside the container
    for index, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                if "citations" in message:
                    for cit_num, cit in enumerate(message["citations"], start=1):
                        with st.expander(f"📍 {_format_citation_label(cit, cit_num)}"):
                            st.info(f"**Excerpt:**\n\n'{cit['excerpt']}'")

                with st.popover("🔄 Regenerate"):
                    st.caption("Leave blank to regenerate with same question. Edit to rephrase.")
                    revised_query = st.text_input(
                        "Edit question (optional):",
                        key=f"edit_field_{index}"
                    )

                    if st.button("Regenerate", key=f"regen_btn_{index}"):
                        regen_payload = {
                            "conversation_id": st.session_state.conversation_id,
                            "question_id": message.get("question_id"),
                            "edited_question": revised_query if revised_query.strip() != "" else None,
                            "created_by": {"analyst_name": "User", "email_id": "user@example.com"}
                        }

                        with st.spinner("Regenerating..."):
                            try:
                                res = requests.post(
                                    f"{BACKEND_URL}/api/v1/chat/regenerate",
                                    json=regen_payload,
                                    headers=HEADERS,
                                )
                                if res.status_code == 200:
                                    res_json = res.json()
                                    if res_json.get("status"):
                                        updated_item = res_json["item"]
                                        st.session_state.chat_history[index]["content"] = updated_item["summary"]
                                        st.session_state.chat_history[index]["citations"] = updated_item.get("citations", [])
                                        if regen_payload["edited_question"] is not None and index > 0:
                                            st.session_state.chat_history[index - 1]["content"] = regen_payload["edited_question"]
                                        st.rerun()
                                    else:
                                        st.error(f"Failed: {res_json.get('message')}")
                                else:
                                    st.error(f"Error: Status {res.status_code}")
                            except Exception as e:
                                st.error(f"Connection error: {e}")

# ==============================================================================
# 📥 4. CHAT INPUT
# ==============================================================================
if user_question := st.chat_input("Ask a question about the selected documents..."):

    if not selected_file_ids:
        st.error("Select at least 1 document from the sidebar first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # 🚀 QUICK FIX ADDED HERE: Write new outputs directly into the container context
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_question)

        ask_payload = {
            "question": user_question,
            "file_ids": selected_file_ids,
            "conversation_id": st.session_state.conversation_id,
            "created_by": {"analyst_name": "User", "email_id": "user@example.com"}
        }

        # 🚀 QUICK FIX ADDED HERE: Write assistant block inside the container context
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Processing with RAG pipeline..."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/api/v1/chat/ask",
                            json=ask_payload,
                            headers=HEADERS,
                        )

                        if response.status_code == 200:
                            res_data = response.json()

                            if res_data.get("status"):
                                item = res_data["item"]
                                st.session_state.conversation_id = item["conversation_id"]
                                bot_summary = item["summary"]
                                citations_list = item.get("citations", [])
                                q_id = item.get("question_id")

                                st.markdown(bot_summary)
                                for cit_num, citation in enumerate(citations_list, start=1):
                                    with st.expander(f"📍 {_format_citation_label(citation, cit_num)}"):
                                        st.info(f"**Excerpt:**\n\n'{citation['excerpt']}'")

                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": bot_summary,
                                    "citations": citations_list,
                                    "question_id": q_id,
                                })
                                st.rerun()
                            else:
                                st.error(f"Backend Error: {res_data.get('message')}")
                        else:
                            st.error(f"HTTP Error: Status {response.status_code}")
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
