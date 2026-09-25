"""
AI Answer Evaluator - Premium Admin Client Interface
====================================================
Self-contained Streamlit front-end client interface for the AI Answer Evaluation System.
Connects to OCR (Port 8001) and Evaluation (Port 8002) FastAPI backend services.
Includes an OCR Engine Warmup & Health Diagnostics section (/health, /config).
"""

import json
import os
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import requests
import streamlit as st

# =====================================================
# GLOBAL CONFIGURATION
# =====================================================
OCR_ENDPOINT = "http://localhost:8001"
EVAL_ENDPOINT = "http://localhost:8002"
REQUEST_TIMEOUT = 60

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Answer Assessment System | Admin Console",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================
# PREMIUM ADMIN CSS DESIGN SYSTEM
# =====================================================
INJECTED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.main .block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Top System Navbar */
.admin-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 24px;
}
.admin-navbar-title {
    font-size: 20px;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.5px;
    margin: 0;
}
.admin-navbar-subtitle {
    font-size: 13px;
    color: #94A3B8;
    margin-top: 3px;
}
.admin-status-group {
    display: flex;
    gap: 10px;
}
.system-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    padding: 5px 12px;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.pill-online {
    background: #064E3B;
    color: #34D399;
    border: 1px solid #059669;
}
.pill-ready {
    background: #1E3A8A;
    color: #60A5FA;
    border: 1px solid #2563EB;
}
.pill-warmup {
    background: #78350F;
    color: #FBBF24;
    border: 1px solid #D97706;
}

/* Section Tag and Titles */
.section-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    color: #2563EB;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.admin-section-title {
    font-size: 18px;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #E2E8F0;
}

/* Cards & Dropzone */
.admin-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
}
.dropzone-box {
    border: 2px dashed #94A3B8;
    background-color: #F8FAFC;
    border-radius: 10px;
    padding: 24px;
    text-align: center;
    margin-bottom: 12px;
}
.dropzone-title {
    font-size: 14px;
    font-weight: 700;
    color: #334155;
    margin-bottom: 4px;
}
.dropzone-subtext {
    font-size: 12px;
    color: #64748B;
}

/* Admin Data Table */
.admin-table-wrapper {
    width: 100%;
    overflow-x: auto;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    margin-top: 14px;
    background: #FFFFFF;
}
.admin-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    text-align: left;
}
.admin-table th {
    background-color: #F8FAFC;
    color: #475569;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
    padding: 12px 16px;
    border-bottom: 1px solid #E2E8F0;
}
.admin-table td {
    padding: 14px 16px;
    color: #1E293B;
    border-bottom: 1px solid #F1F5F9;
    vertical-align: middle;
}
.admin-table tr:last-child td {
    border-bottom: none;
}
.admin-table tr:hover td {
    background-color: #F8FAFC;
}

.col-id {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: #64748B;
    width: 40px;
}
.col-concept {
    font-weight: 600;
    color: #0F172A;
    line-height: 1.5;
}
.sim-container {
    display: flex;
    align-items: center;
    gap: 12px;
}
.sim-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    width: 55px;
    color: #0F172A;
}
.sim-track {
    flex-grow: 1;
    height: 8px;
    background: #E2E8F0;
    border-radius: 4px;
    overflow: hidden;
}
.sim-bar {
    height: 100%;
    background: linear-gradient(90deg, #2563EB, #1D4ED8);
    border-radius: 4px;
}

/* Status Badges */
.status-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 6px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    text-align: center;
}
.status-covered {
    background-color: #DCFCE7;
    color: #15803D;
    border: 1px solid #86EFAC;
}
.status-partial {
    background-color: #FEF3C7;
    color: #B45309;
    border: 1px solid #FDE047;
}
.status-missing {
    background-color: #FEE2E2;
    color: #B91C1C;
    border: 1px solid #FCA5A5;
}
.status-neutral {
    background-color: #F1F5F9;
    color: #475569;
    border: 1px solid #CBD5E1;
}

/* KPI Metric Cards */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}
.kpi-title {
    font-size: 11px;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 800;
    color: #0F172A;
    font-family: 'JetBrains Mono', monospace;
}

/* Concept Panels */
.concept-panel {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 18px 20px;
    height: 100%;
}
.concept-panel-header-green {
    font-size: 14px;
    font-weight: 700;
    color: #15803D;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding-bottom: 10px;
    border-bottom: 2px solid #DCFCE7;
    margin-bottom: 12px;
}
.concept-panel-header-red {
    font-size: 14px;
    font-weight: 700;
    color: #B91C1C;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding-bottom: 10px;
    border-bottom: 2px solid #FEE2E2;
    margin-bottom: 12px;
}

/* Executive Callout Boxes */
.callout-error {
    background-color: #FEF2F2;
    border-left: 4px solid #EF4444;
    border-radius: 8px;
    padding: 16px 20px;
    margin-top: 20px;
    color: #991B1B;
}
.callout-error-title {
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 6px;
    color: #991B1B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.callout-suggestion {
    background-color: #EFF6FF;
    border-left: 4px solid #2563EB;
    border-radius: 8px;
    padding: 16px 20px;
    margin-top: 20px;
    color: #1E40AF;
}
.callout-suggestion-title {
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 6px;
    color: #1E40AF;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Footer Text */
.admin-footer {
    text-align: center;
    color: #94A3B8;
    font-size: 12px;
    font-weight: 500;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #E2E8F0;
}

/* Graph Container Styling */
.graph-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
}
.graph-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #F1F5F9;
}
.graph-title {
    font-size: 16px;
    font-weight: 700;
    color: #0F172A;
}
.graph-axis-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    background: #EFF6FF;
    color: #2563EB;
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid #BFDBFE;
}

/* Streamlit Button Overrides */
div.stButton > button {
    border-radius: 8px;
    font-weight: 700;
    font-size: 14px;
    padding: 0.65rem 1.4rem;
    transition: all 0.15s ease-in-out;
}
</style>
"""

st.markdown(INJECTED_CSS, unsafe_allow_html=True)


# =====================================================
# SYSTEM HEALTH & WARMUP HELPERS (/health, /config)
# =====================================================
def fetch_health(endpoint: str) -> Dict[str, Any]:
    """Fetches system health status from /health endpoint."""
    url = f"{endpoint.rstrip('/')}/health"
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            return resp.json()
        return {"status": f"HTTP {resp.status_code}", "detail": resp.text[:100]}
    except Exception as e:
        return {"status": "UNREACHABLE", "detail": str(e)}


def fetch_config(endpoint: str) -> Dict[str, Any]:
    """Fetches configuration details from /config endpoint."""
    url = f"{endpoint.rstrip('/')}/config"
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            return resp.json()
        return {"status": f"HTTP {resp.status_code}", "detail": "No explicit /config endpoint or offline"}
    except Exception as e:
        return {"status": "UNAVAILABLE", "detail": str(e)}


def run_single_warmup_ocr() -> Tuple[bool, float, str]:
    """
    Executes a single continuous warmup request targeting the OCR engine
    using the specific file 'imgs/test17.jpg'.
    """
    candidates = [
        os.path.join("imgs", "test17.jpg"),
        os.path.join("imgs", "test17.jpeg"),
        "imgs/test17.jpg",
        "imgs/test17.jpeg",
        "c:/Users/User/code_file_folder/python/imgs/test17.jpg",
        "c:/Users/User/code_file_folder/python/imgs/test17.jpeg",
    ]

    img_path = None
    for cand in candidates:
        if os.path.exists(cand):
            img_path = cand
            break

    if not img_path:
        return False, 0.0, "Warmup target image 'imgs/test17.jpg' not found."

    url = f"{OCR_ENDPOINT.rstrip('/')}/ocr"
    try:
        start_t = time.perf_counter()
        with open(img_path, "rb") as f:
            files = {"file": (os.path.basename(img_path), f.read(), "image/jpeg")}
            response = requests.post(url, files=files, timeout=None)
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        if response.status_code != 200:
            return False, latency_ms, f"HTTP {response.status_code}"

        json_data = response.json()
        text = ""
        if isinstance(json_data, dict):
            for k in ["text", "extracted_text", "reconstructed", "content", "result"]:
                if k in json_data and json_data[k]:
                    text = str(json_data[k])
                    break
        return True, latency_ms, f"Success ({len(text)} chars extracted)"
    except Exception as e:
        return False, 0.0, str(e)


# =====================================================
# UI COMPONENTS & PIPELINE SERVICES
# =====================================================
def render_header() -> None:
    """Renders top system administration header with live gateway status indicators."""
    warmup_active = st.session_state.get("warmup_running", False)
    status_pill_html = (
        '<span class="system-pill pill-warmup">WARMUP: ACTIVE</span>'
        if warmup_active
        else '<span class="system-pill pill-ready">GATEWAYS: ACTIVE</span>'
    )

    st.markdown(
        f"""
        <div class="admin-navbar">
            <div>
                <h1 class="admin-navbar-title">AI ANSWER ASSESSMENT SYSTEM</h1>
                <div class="admin-navbar-subtitle">Enterprise Academic Assessment & Semantic Evaluation Console</div>
            </div>
            <div class="admin-status-group">
                <span class="system-pill pill-online">SYSTEM: ONLINE</span>
                {status_pill_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_warmup_graph() -> None:
    """
    Renders an attractive performance graph representing:
    X-axis: CYCLES COMPLETED
    Y-axis: LAST LATENCY (ms)
    """
    history = st.session_state.get("warmup_history", [])

    # Sync history if warmup_count is positive but history array was uninitialized
    if st.session_state.get("warmup_count", 0) > 0 and not history:
        history = [{
            "CYCLES COMPLETED": st.session_state.get("warmup_count", 0),
            "LAST LATENCY (ms)": round(st.session_state.get("warmup_last_latency", 0.0), 1),
        }]
        st.session_state["warmup_history"] = history

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="graph-card">
            <div class="graph-header">
                <div>
                    <div class="section-tag" style="font-size: 10px; margin-bottom: 2px;">PERFORMANCE BENCHMARK GRAPH</div>
                    <div class="graph-title">Warmup Latency vs. Cycles Completed</div>
                </div>
                <div class="graph-axis-tag">X: CYCLES COMPLETED | Y: LAST LATENCY (ms)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if history:
        df = pd.DataFrame(history)

        col1, col2, col3, col4 = st.columns(4)
        latest_lat = history[-1]["LAST LATENCY (ms)"]
        total_cycles = history[-1]["CYCLES COMPLETED"]
        valid_lats = [h["LAST LATENCY (ms)"] for h in history if h["LAST LATENCY (ms)"] > 0]
        min_lat = min(valid_lats) if valid_lats else 0.0
        avg_lat = sum(valid_lats) / len(valid_lats) if valid_lats else 0.0

        with col1:
            st.metric("Cycles Completed (X)", f"{total_cycles}")
        with col2:
            st.metric("Latest Latency (Y)", f"{latest_lat:.1f} ms")
        with col3:
            st.metric("Best Latency", f"{min_lat:.1f} ms")
        with col4:
            st.metric("Avg Latency", f"{avg_lat:.1f} ms")

        st.markdown("<br>", unsafe_allow_html=True)

        tab_line, tab_scatter, tab_data = st.tabs(["📈 Line Graph", "🟢 Scatter Plot", "📋 History Log"])

        with tab_line:
            st.line_chart(
                df,
                x="CYCLES COMPLETED",
                y="LAST LATENCY (ms)",
                color="#2563EB",
                use_container_width=True,
            )

        with tab_scatter:
            st.scatter_chart(
                df,
                x="CYCLES COMPLETED",
                y="LAST LATENCY (ms)",
                color="#2563EB",
                size=12,
                use_container_width=True,
            )

        with tab_data:
            st.dataframe(df, use_container_width=True, height=200)

    else:
        st.info(
            "📊 **No Warmup Benchmark Data Yet**\n\n"
            "Click **START WARMUP** above to run continuous inference iterations. "
            "The chart will automatically update with **CYCLES COMPLETED on the X-axis** and **LAST LATENCY (ms) on the Y-axis**."
        )


def render_warmup_section() -> None:
    """
    Renders the Warming Up section for continuous OCR model warming using imgs/test17.jpg,
    along with system status & health indicators (/health, /config) and performance graph.
    """
    st.markdown('<div class="section-tag">SYSTEM DIAGNOSTICS & WARMUP</div>', unsafe_allow_html=True)
    st.markdown('<div class="admin-section-title">Warming Up</div>', unsafe_allow_html=True)

    warmup_running = st.session_state["warmup_running"]

    with st.expander("OCR Engine Warmup & Health Diagnostics (/health, /config)", expanded=warmup_running):
        col_ctl, col_diag = st.columns([1, 1])

        with col_ctl:
            st.markdown("#### OCR Warmup Control")
            st.markdown(
                "Runs continuous OCR inference requests on target image `imgs/test17.jpg` to warm up PyTorch/TrOCR model tensors in GPU memory."
            )

            if warmup_running:
                badge_html = '<span class="status-badge status-partial">IN PROGRESS - WARMING UP OCR</span>'
                btn_label = "STOP WARMUP"
                btn_type = "secondary"
            else:
                badge_html = '<span class="status-badge status-covered">READY FOR PIPELINE USE</span>'
                btn_label = "START WARMUP"
                btn_type = "primary"

            st.markdown(f"**Warmup Status:** {badge_html}", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button(btn_label, type=btn_type, key="warmup_toggle_btn", use_container_width=True):
                st.session_state["warmup_running"] = not st.session_state["warmup_running"]
                st.rerun()

            st.markdown(
                f"""
                <div class="kpi-card" style="margin-top: 16px;">
                    <div class="kpi-title">Warmup Benchmark Image Target</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; color: #2563EB;">imgs/test17.jpg</div>
                    <div style="margin-top: 10px; font-size: 12px; color: #475569; font-family: 'JetBrains Mono', monospace;">
                        <strong>CYCLES COMPLETED:</strong> {st.session_state['warmup_count']}<br>
                        <strong>LAST LATENCY:</strong> {st.session_state['warmup_last_latency']:.1f} ms<br>
                        <strong>STATUS DETAILS:</strong> {st.session_state['warmup_last_msg']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_diag:
            st.markdown("#### OCR Endpoint Status (/health, /config)")

            ocr_health = fetch_health(OCR_ENDPOINT)
            ocr_config = fetch_config(OCR_ENDPOINT)

            st.markdown("**OCR Endpoint Health (`/health`):**")
            st.json(ocr_health, expanded=False)

            st.markdown("**OCR Configuration (`/config`):**")
            st.json(ocr_config, expanded=False)

        # Visually attractive Latency vs Cycles Graph
        render_warmup_graph()

    # Active Warmup Loop Execution
    if st.session_state["warmup_running"]:
        success, latency, msg = run_single_warmup_ocr()
        st.session_state["warmup_count"] += 1
        st.session_state["warmup_last_latency"] = latency
        st.session_state["warmup_last_msg"] = msg if success else f"Error: {msg}"
        st.session_state["warmup_history"].append({
            "CYCLES COMPLETED": st.session_state["warmup_count"],
            "LAST LATENCY (ms)": round(latency, 1),
        })
        time.sleep(15)
        st.rerun()


def render_upload_section() -> Optional[Any]:
    """Renders the image ingestion upload card."""
    st.markdown('<div class="section-tag">INPUT PHASE 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="admin-section-title">Answer Sheet Image Ingestion</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="dropzone-box">
            <div class="dropzone-title">Upload High-Resolution Answer Sheet Image</div>
            <div class="dropzone-subtext">Supported File Formats: PNG, JPG, JPEG</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Answer Sheet Image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        key="image_uploader",
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        with col2:
            st.markdown(
                f"""
                <div style="padding-top: 10px;">
                    <span class="status-badge status-covered">IMAGE LOADED</span>
                    <p style="margin-top: 10px; font-size: 13px; color: #475569; font-family: 'JetBrains Mono', monospace;">
                        <strong>FILE:</strong> {uploaded_file.name}<br>
                        <strong>SIZE:</strong> {uploaded_file.size / 1024:.1f} KB
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    return uploaded_file


def render_reference_and_question() -> Tuple[str, str]:
    """Renders the reference answer and target question specification inputs."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-tag">INPUT PHASE 02</div>', unsafe_allow_html=True)
        st.markdown('<div class="admin-section-title">Target Question Specification</div>', unsafe_allow_html=True)
        question = st.text_area(
            "Target Question",
            height=140,
            placeholder="Enter the official question prompt...",
            label_visibility="collapsed",
            key="question_input",
        )

    with col2:
        st.markdown('<div class="section-tag">INPUT PHASE 03</div>', unsafe_allow_html=True)
        st.markdown('<div class="admin-section-title">Reference Model Answer</div>', unsafe_allow_html=True)
        ref_answer = st.text_area(
            "Reference Answer",
            height=140,
            placeholder="Enter or paste the authoritative benchmark answer...",
            label_visibility="collapsed",
            key="ref_answer_input",
        )

    return ref_answer, question


def call_ocr(image_file: Any) -> Optional[str]:
    """Sends the uploaded image to the OCR endpoint and extracts student answer text."""
    url = f"{OCR_ENDPOINT.rstrip('/')}/ocr"
    try:
        image_bytes = image_file.getvalue()
        files = {"file": (image_file.name, image_bytes, image_file.type or "image/jpeg")}
        response = requests.post(url, files=files, timeout=None)

        if response.status_code != 200:
            st.error(f"OCR Request Failed (HTTP {response.status_code}): {response.text}")
            return None

        json_data = response.json()
        extracted_text = None

        if isinstance(json_data, dict):
            for key in ["text", "extracted_text", "reconstructed", "content", "result"]:
                if key in json_data and json_data[key]:
                    extracted_text = str(json_data[key])
                    break

            if extracted_text is None:
                for k, v in json_data.items():
                    if isinstance(v, str) and v.strip():
                        extracted_text = v
                        break

        if not extracted_text:
            st.error(f"OCR response missing expected text field. Available response keys: {list(json_data.keys()) if isinstance(json_data, dict) else type(json_data)}")
            return None

        return extracted_text

    except requests.exceptions.ConnectionError:
        st.error(f"Failed to connect to OCR service at '{url}'. Please verify OCR endpoint status.")
        return None
    except Exception as e:
        st.error(f"Unexpected error during OCR text extraction: {str(e)}")
        return None


def call_evaluate(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Sends the evaluation JSON payload to the Evaluation FastAPI endpoint."""
    url = f"{EVAL_ENDPOINT.rstrip('/')}/evaluate"
    try:
        response = requests.post(url, json=payload, timeout=None)

        if response.status_code != 200:
            st.error(f"Evaluation Request Failed (HTTP {response.status_code}): {response.text}")
            return None

        return response.json()

    except requests.exceptions.ConnectionError:
        st.error(f"Failed to connect to Evaluation service at '{url}'. Please verify Evaluation endpoint status.")
        return None
    except Exception as e:
        st.error(f"Unexpected error during answer evaluation: {str(e)}")
        return None


def render_concept_table(concepts: List[Dict[str, Any]]) -> None:
    """
    Renders the concept breakdown table cleanly without HTML indentation issues.
    Fixes raw HTML wrapping code blocks in Streamlit.
    """
    if not concepts:
        st.info("No concept breakdown records available.")
        return

    rows_list = []
    for idx, item in enumerate(concepts, start=1):
        if isinstance(item, dict):
            concept_name = str(item.get("concept", f"Concept Requirement {idx}"))
            sim_val = item.get("similarity", 0.0)
            status_raw = str(item.get("status", "")).lower()
        else:
            concept_name = str(item)
            sim_val = 0.0
            status_raw = "unknown"

        try:
            sim_float = float(sim_val)
            sim_pct = sim_float * 100.0 if sim_float <= 1.0 else sim_float
            sim_str = f"{sim_pct:.1f}%"
        except (ValueError, TypeError):
            sim_float = 0.0
            sim_pct = 0.0
            sim_str = str(sim_val)

        if status_raw == "covered":
            pill_html = '<span class="status-badge status-covered">COVERED</span>'
        elif status_raw == "partial":
            pill_html = '<span class="status-badge status-partial">PARTIAL</span>'
        elif status_raw == "missing":
            pill_html = '<span class="status-badge status-missing">MISSING</span>'
        else:
            pill_html = f'<span class="status-badge status-neutral">{status_raw.upper()}</span>'

        pct_bar_width = min(max(sim_pct, 0.0), 100.0)

        row_html = (
            f'<tr>'
            f'<td class="col-id">{idx:02d}</td>'
            f'<td class="col-concept">{concept_name}</td>'
            f'<td style="width: 220px;">'
            f'<div class="sim-container">'
            f'<span class="sim-text">{sim_str}</span>'
            f'<div class="sim-track"><div class="sim-bar" style="width: {pct_bar_width:.1f}%;"></div></div>'
            f'</div>'
            f'</td>'
            f'<td style="width: 130px;" class="col-status">{pill_html}</td>'
            f'</tr>'
        )
        rows_list.append(row_html)

    table_rows_str = "".join(rows_list)

    full_table_html = (
        f'<div class="admin-table-wrapper">'
        f'<table class="admin-table">'
        f'<thead>'
        f'<tr>'
        f'<th style="width: 50px;">#</th>'
        f'<th>Benchmark Concept Sentence</th>'
        f'<th style="width: 220px;">Semantic Match</th>'
        f'<th style="width: 130px;">Coverage Status</th>'
        f'</tr>'
        f'</thead>'
        f'<tbody>'
        f'{table_rows_str}'
        f'</tbody>'
        f'</table>'
        f'</div>'
    )

    st.markdown(full_table_html, unsafe_allow_html=True)


def render_report(result: Dict[str, Any]) -> None:
    """Renders the executive evaluation report and data metrics."""
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-tag">SYSTEM OUTPUT</div>', unsafe_allow_html=True)
    st.markdown('<div class="admin-section-title">Evaluation & Assessment Report</div>', unsafe_allow_html=True)

    llm_error = result.get("llm_error")
    if llm_error:
        st.error(f"LLM Analytics Warning: {llm_error}")

    score = float(result.get("score", 0.0))
    similarity = float(result.get("similarity", 0.0))
    concept_coverage = float(result.get("concept_coverage", 0.0))
    concepts = result.get("concepts", [])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        score_display = f"{score:.1f}/100" if not score.is_integer() else f"{int(score)}/100"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Overall Grade Score</div>
                <div class="kpi-value">{score_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Semantic Similarity</div>
                <div class="kpi-value">{similarity * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Concept Coverage Rate</div>
                <div class="kpi-value">{concept_coverage * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        if score >= 80:
            status_badge_html = '<span class="status-badge status-covered">PASSED: EXCELLENT</span>'
        elif score >= 50:
            status_badge_html = '<span class="status-badge status-partial">NEEDS IMPROVEMENT</span>'
        else:
            status_badge_html = '<span class="status-badge status-missing">CRITICAL DEFICIT</span>'

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Assessment Classification</div>
                <div style="padding-top: 6px;">{status_badge_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    progress_val = min(max(score / 100.0, 0.0), 1.0)
    st.progress(progress_val)

    # Render Concept Breakdown Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Concept Breakdown Analysis")
    render_concept_table(concepts)

    # Render Covered vs Missing Concepts Grid
    report_details = result.get("report", {}) or {}
    correct_concepts = report_details.get("correct_concepts", [])
    missing_concepts = report_details.get("missing_concepts", [])

    st.markdown("<br>", unsafe_allow_html=True)
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown(
            """
            <div class="concept-panel">
                <div class="concept-panel-header-green">Identified Covered Concepts</div>
            """,
            unsafe_allow_html=True,
        )
        if correct_concepts:
            for item in correct_concepts:
                st.markdown(f"- **[COVERED]** {item}")
        else:
            st.markdown("*No covered concepts identified.*")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown(
            """
            <div class="concept-panel">
                <div class="concept-panel-header-red">Identified Missing / Deficit Concepts</div>
            """,
            unsafe_allow_html=True,
        )
        if missing_concepts:
            for item in missing_concepts:
                st.markdown(f"- **[MISSING]** {item}")
        else:
            st.markdown("*No concept deficits identified.*")
        st.markdown("</div>", unsafe_allow_html=True)

    # Errors Warning Callout
    errors = report_details.get("errors", [])
    if errors:
        error_items = "".join([f"<li style='margin-bottom: 4px;'>{err}</li>" for err in errors])
        st.markdown(
            f"""
            <div class="callout-error">
                <div class="callout-error-title">Detected Conceptual Errors & Anomalies</div>
                <ul style="margin-top: 6px; margin-bottom: 0; padding-left: 20px;">
                    {error_items}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Suggestion Callout Box
    improvement = report_details.get("improvement", "")
    if improvement:
        st.markdown(
            f"""
            <div class="callout-suggestion">
                <div class="callout-suggestion-title">Strategic Improvement Recommendation</div>
                <div style="font-size: 14px; line-height: 1.6;">{improvement}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    """Renders footer bar."""
    st.markdown(
        '<div class="admin-footer">AI Answer Assessment Engine • Enterprise System v1.0</div>',
        unsafe_allow_html=True,
    )


# =====================================================
# PIPELINE ORCHESTRATION & MAIN CONTROLLER
# =====================================================
def main() -> None:
    if "stage" not in st.session_state:
        st.session_state["stage"] = "idle"
    if "ocr_text" not in st.session_state:
        st.session_state["ocr_text"] = ""
    if "confirmed_text" not in st.session_state:
        st.session_state["confirmed_text"] = ""
    if "eval_result" not in st.session_state:
        st.session_state["eval_result"] = None

    # Warmup state initialization
    if "warmup_running" not in st.session_state:
        st.session_state["warmup_running"] = False
    if "warmup_count" not in st.session_state:
        st.session_state["warmup_count"] = 0
    if "warmup_last_latency" not in st.session_state:
        st.session_state["warmup_last_latency"] = 0.0
    if "warmup_last_msg" not in st.session_state:
        st.session_state["warmup_last_msg"] = "Idle"
    if "warmup_history" not in st.session_state:
        st.session_state["warmup_history"] = []

    render_header()
    render_warmup_section()

    uploaded_file = render_upload_section()
    ref_answer, question = render_reference_and_question()

    st.markdown("<br>", unsafe_allow_html=True)
    submit_clicked = st.button("EXECUTE EVALUATION PIPELINE", use_container_width=True, type="primary")

    # Handle Initial Submission (Runs OCR AND Evaluation automatically without manual proceed step)
    if submit_clicked:
        missing_inputs = []
        if uploaded_file is None:
            missing_inputs.append("Answer Sheet Image")
        if not ref_answer.strip():
            missing_inputs.append("Reference Answer Specification")
        if not question.strip():
            missing_inputs.append("Target Question Specification")

        if missing_inputs:
            st.error("Submission Error: The following required inputs are missing:\n- " + "\n- ".join(missing_inputs))
        else:
            # Step 1: Run OCR
            with st.spinner("Extracting handwritten text via OCR engine..."):
                extracted_text = call_ocr(uploaded_file)

            if extracted_text is not None:
                st.session_state["ocr_text"] = extracted_text
                st.session_state["confirmed_text"] = extracted_text
                st.session_state["saved_question"] = question.strip()
                st.session_state["saved_ref_answer"] = ref_answer.strip()

                # Step 2: IMMEDIATELY Run Evaluation (Automated — no wait/proceed required)
                payload = {
                    "question": st.session_state["saved_question"],
                    "reference_answer": st.session_state["saved_ref_answer"],
                    "student_answer": extracted_text,
                }

                with st.spinner("Processing semantic evaluation and concept coverage..."):
                    eval_result = call_evaluate(payload)

                if eval_result is not None:
                    st.session_state["eval_result"] = eval_result
                    st.session_state["stage"] = "report_ready"
                    st.rerun()

    # Step 3: Extracted Text Review & Edit Panel (Optional for user re-runs)
    if st.session_state["stage"] in ["ocr_done", "report_ready"]:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Extracted Student Answer Text (Click to inspect or edit)", expanded=False):
            edited_student_answer = st.text_area(
                "Extracted Text Content",
                value=st.session_state["confirmed_text"],
                height=150,
                key="student_answer_edit_area",
            )

            re_eval_clicked = st.button("Re-evaluate with Modified Text", use_container_width=True)

            if re_eval_clicked:
                if not edited_student_answer.strip():
                    st.warning("Student answer text cannot be empty.")
                else:
                    st.session_state["confirmed_text"] = edited_student_answer.strip()

                    payload = {
                        "question": st.session_state.get("saved_question", question.strip()),
                        "reference_answer": st.session_state.get("saved_ref_answer", ref_answer.strip()),
                        "student_answer": st.session_state["confirmed_text"],
                    }

                    with st.spinner("Re-evaluating modified text..."):
                        eval_result = call_evaluate(payload)

                    if eval_result is not None:
                        st.session_state["eval_result"] = eval_result
                        st.session_state["stage"] = "report_ready"
                        st.rerun()

    # Step 4: Display Evaluation Report
    if st.session_state["stage"] == "report_ready" and st.session_state["eval_result"] is not None:
        render_report(st.session_state["eval_result"])

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("RESET ASSESSMENT CONSOLE", use_container_width=True):
            st.session_state["stage"] = "idle"
            st.session_state["ocr_text"] = ""
            st.session_state["confirmed_text"] = ""
            st.session_state["eval_result"] = None
            st.rerun()

    render_footer()


if __name__ == "__main__":
    main()
