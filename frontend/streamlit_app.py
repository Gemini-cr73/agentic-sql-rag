from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# -----------------------------
# Paths / logo loading
# -----------------------------
APP_FILE = Path(__file__).resolve()
PROJECT_ROOT = APP_FILE.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"


def _has_logo() -> bool:
    return LOGO_PATH.exists() and LOGO_PATH.is_file()


st.set_page_config(
    page_title="Agentic SQL RAG",
    page_icon="🔎",
    layout="wide",
)


# -----------------------------
# Helpers
# -----------------------------
def _normalize_base_url(url: str) -> str:
    return (url or DEFAULT_API_BASE_URL).strip().rstrip("/")


def _post_json(
    url: str, payload: dict[str, Any], timeout: int = 90
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as e:
        return None, f"Request failed: {e}"

    try:
        body = response.json()
    except ValueError:
        body = {"raw_text": response.text}

    if response.status_code >= 400:
        return None, f"HTTP {response.status_code}: {json.dumps(body, indent=2)}"

    return body, None


def _get_json(url: str, timeout: int = 30) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        return None, f"Request failed: {e}"

    try:
        body = response.json()
    except ValueError:
        body = {"raw_text": response.text}

    if response.status_code >= 400:
        return None, f"HTTP {response.status_code}: {json.dumps(body, indent=2)}"

    return body, None


def _post_file(
    url: str,
    file_name: str,
    file_bytes: bytes,
    mime_type: str = "text/plain",
    timeout: int = 180,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.post(
            url,
            files={"file": (file_name, file_bytes, mime_type)},
            timeout=timeout,
        )
    except requests.RequestException as e:
        return None, f"Request failed: {e}"

    try:
        body = response.json()
    except ValueError:
        body = {"raw_text": response.text}

    if response.status_code >= 400:
        return None, f"HTTP {response.status_code}: {json.dumps(body, indent=2)}"

    return body, None


def _get_health(base_url: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json(), None
    except Exception as e:
        return None, str(e)


def _grounding_badge(grounding: dict[str, Any]) -> tuple[str, str]:
    score = float(grounding.get("grounding_score", 0.0))
    hallucinations = bool(grounding.get("has_hallucinations", True))

    if score >= 0.85 and not hallucinations:
        return "🟢 Strong grounding", f"Grounding score: {score:.2f}"
    if score >= 0.50:
        return "🟡 Partial grounding", f"Grounding score: {score:.2f}"
    return "🔴 Weak grounding", f"Grounding score: {score:.2f}"


def _render_grounding_badge_inline(grounding: dict[str, Any]) -> None:
    if not grounding:
        st.caption("Grounding: not available")
        return

    label, detail = _grounding_badge(grounding)
    st.caption(f"{label} · {detail}")


def _render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        st.caption("No citations returned.")
        return

    st.markdown("**Citations**")
    for c in citations:
        citation_id = c.get("id", "?")
        title = c.get("title") or "Untitled chunk"
        stable_id = c.get("stable_id") or "unknown"
        source = c.get("source") or "n/a"
        char_start = c.get("char_start")
        char_end = c.get("char_end")

        line = (
            f"- **[{citation_id}]** {title}  \n"
            f"  source: `{source}` · stable_id: `{stable_id}`"
        )

        if char_start is not None and char_end is not None:
            line += f" · chars: `{char_start}-{char_end}`"

        st.markdown(line)


def _render_evidence_cards(
    citations: list[dict[str, Any]], grounding: dict[str, Any], raw_json: dict[str, Any]
) -> None:
    if not citations:
        st.caption("No evidence cards available.")
        return

    st.markdown("**Evidence**")

    support_by_chunk: dict[str, float] = {}
    for item in grounding.get("sentence_checks", []):
        sid = item.get("best_support_stable_id")
        if sid is None:
            continue
        support_by_chunk[str(sid)] = max(
            float(item.get("best_support_score", 0.0)),
            support_by_chunk.get(str(sid), 0.0),
        )

    retrieved_rows = raw_json.get("retrieved", [])
    retrieved_by_sid: dict[str, dict[str, Any]] = {}
    if isinstance(retrieved_rows, list):
        for row in retrieved_rows:
            if isinstance(row, dict) and row.get("stable_id") is not None:
                retrieved_by_sid[str(row["stable_id"])] = row

    for idx, c in enumerate(citations, start=1):
        stable_id = str(c.get("stable_id") or "unknown")
        title = c.get("title") or f"Evidence {idx}"
        source = c.get("source") or "n/a"
        row = retrieved_by_sid.get(stable_id, {})
        support_score = support_by_chunk.get(stable_id)

        with st.expander(f"Evidence {idx} — {title}"):
            st.write(f"**Source:** {source}")
            st.write(
                f"**stable_id:** `{stable_id}` · **doc_id:** `{c.get('doc_id')}` · "
                f"**chunk_id:** `{c.get('chunk_id')}`"
            )
            st.write(f"**char range:** `{c.get('char_start')}` - `{c.get('char_end')}`")

            if support_score is not None:
                st.caption(f"Best support score: {support_score:.4f}")

            if row:
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Base Score",
                    f"{float(row.get('base_score', 0.0)):.4f}"
                    if row.get("base_score") is not None
                    else "n/a",
                )
                c2.metric(
                    "Rerank Score",
                    f"{float(row.get('rerank_score', 0.0)):.4f}"
                    if row.get("rerank_score") is not None
                    else "n/a",
                )
                c3.metric(
                    "Final Score",
                    f"{float(row.get('final_score', 0.0)):.4f}"
                    if row.get("final_score") is not None
                    else "n/a",
                )

                if row.get("rerank_method"):
                    st.caption(f"Rerank method: {row.get('rerank_method')}")


def _render_rerank_summary(raw_json: dict[str, Any]) -> None:
    retrieved = raw_json.get("retrieved", [])
    if not isinstance(retrieved, list) or not retrieved:
        st.caption("Rerank summary: not available")
        return

    reranked = [
        r
        for r in retrieved
        if isinstance(r, dict) and r.get("rerank_score") is not None
    ]
    if not reranked:
        st.caption("Rerank summary: rerank data not returned")
        return

    top_rows = sorted(
        reranked,
        key=lambda r: float(r.get("final_score", 0.0)),
        reverse=True,
    )[:3]

    summary_parts = [
        f"{row.get('stable_id')} ({float(row.get('final_score', 0.0)):.3f})"
        for row in top_rows
    ]
    st.caption(f"Rerank summary: top results after rerank → {', '.join(summary_parts)}")


def _render_agent_tools(tools: list[dict[str, Any]]) -> None:
    if not tools:
        st.caption("No agent tool trace returned.")
        return

    st.markdown("**Tool Trace**")
    for i, tool in enumerate(tools, start=1):
        with st.expander(f"Tool Call {i}: {tool.get('name', 'unknown')}"):
            st.json(tool)


def _build_ask_payload(
    query: str,
    retrieval_mode: str,
    alpha: float,
    k_final: int,
    rerank: bool,
    rerank_method: str,
    rerank_weight: float,
    rerank_top_k: int,
    max_sentences: int,
    support_threshold: float,
) -> dict[str, Any]:
    return {
        "query": query,
        "mode": retrieval_mode,
        "alpha": alpha,
        "k_final": k_final,
        "rerank": rerank,
        "rerank_method": rerank_method,
        "rerank_weight": rerank_weight,
        "rerank_top_k": rerank_top_k,
        "max_sentences": max_sentences,
        "support_threshold": support_threshold,
    }


def _build_agent_payload(
    query: str,
    retrieval_mode: str,
    alpha: float,
    k_final: int,
) -> dict[str, Any]:
    return {
        "query": query,
        "mode": retrieval_mode,
        "alpha": alpha,
        "k_final": k_final,
        "use_agent_memory": False,
        "max_iterations": 3,
    }


def _append_user_message(content: str) -> None:
    st.session_state["messages"].append({"role": "user", "content": content})


def _append_assistant_message(
    content: str,
    citations: list[dict[str, Any]] | None = None,
    grounding: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    raw_json: dict[str, Any] | None = None,
) -> None:
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": content or "",
            "citations": citations or [],
            "grounding": grounding or {},
            "tools": tools or [],
            "raw_json": raw_json or {},
        }
    )


def _render_assistant_message(msg: dict[str, Any], show_raw_json: bool) -> None:
    st.markdown(msg.get("content", ""))

    grounding = msg.get("grounding", {}) or {}
    citations = msg.get("citations", []) or []
    raw_json = msg.get("raw_json", {}) or {}
    tools = msg.get("tools", []) or []

    _render_grounding_badge_inline(grounding)
    _render_rerank_summary(raw_json)

    if citations:
        _render_citations(citations)
        _render_evidence_cards(citations, grounding, raw_json)

    if tools:
        _render_agent_tools(tools)

    if show_raw_json and raw_json:
        with st.expander("Raw JSON"):
            st.json(raw_json)


def _fetch_evaluation_summary(url: str) -> tuple[dict[str, Any] | None, str | None]:
    data, err = _get_json(url)
    if err:
        return None, err
    if not isinstance(data, dict):
        return None, "Invalid evaluation summary response format."
    return data, None


def _fetch_evaluation_runs(url: str) -> tuple[list[dict[str, Any]], str | None]:
    data, err = _get_json(url)
    if err:
        return [], err
    runs = data.get("runs", []) if isinstance(data, dict) else []
    if not isinstance(runs, list):
        return [], "Invalid evaluation runs response format."
    return runs, None


def _render_evaluation_dashboard(
    summary: dict[str, Any] | None,
    runs: list[dict[str, Any]],
    summary_err: str | None,
    runs_err: str | None,
) -> None:
    st.subheader("Evaluation Dashboard")

    if summary_err:
        st.warning(f"Evaluation summary unavailable: {summary_err}")
    elif not summary or not summary.get("available", False):
        st.info("No evaluation summary available yet. Generate metrics.json first.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Precision@K", f"{float(summary.get('precision_at_k', 0.0)):.3f}")
        c2.metric("Recall@K", f"{float(summary.get('recall_at_k', 0.0)):.3f}")
        c3.metric("MRR", f"{float(summary.get('mrr', 0.0)):.3f}")
        c4.metric("nDCG", f"{float(summary.get('ndcg', 0.0)):.3f}")
        c5.metric(
            "Grounding",
            f"{float(summary.get('average_grounding_score', 0.0)):.3f}",
        )

        rerank_comp = summary.get("rerank_comparison", {})
        if isinstance(rerank_comp, dict) and rerank_comp:
            st.markdown("**Rerank Comparison**")
            st.json(rerank_comp)

        st.caption(
            f"Queries evaluated: {int(summary.get('query_count', 0))} · source: {summary.get('source', '')}"
        )

    st.markdown("---")
    st.markdown("**Query-by-Query Analysis**")

    if runs_err:
        st.warning(f"Evaluation runs unavailable: {runs_err}")
    elif not runs:
        st.info("No query-by-query evaluation data available yet.")
    else:
        for idx, run in enumerate(runs, start=1):
            query_text = run.get("query", "") or f"Run {idx}"
            with st.expander(f"Run {idx} — {query_text}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Precision@K", f"{float(run.get('precision_at_k', 0.0)):.3f}")
                c2.metric("Recall@K", f"{float(run.get('recall_at_k', 0.0)):.3f}")
                c3.metric("MRR", f"{float(run.get('mrr', 0.0)):.3f}")
                c4.metric("nDCG", f"{float(run.get('ndcg', 0.0)):.3f}")

                c5, c6, c7 = st.columns(3)
                c5.metric(
                    "Grounding",
                    f"{float(run.get('grounding_score', 0.0)):.3f}",
                )
                c6.metric(
                    "Rerank Enabled",
                    "Yes" if bool(run.get("rerank_enabled", False)) else "No",
                )
                c7.metric(
                    "Latency (ms)",
                    f"{float(run.get('latency_ms', 0.0)):.1f}",
                )

                if run.get("notes"):
                    st.caption(f"Notes: {run.get('notes')}")


# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "last_upload_status" not in st.session_state:
    st.session_state["last_upload_status"] = None

if "uploader_reset_counter" not in st.session_state:
    st.session_state["uploader_reset_counter"] = 0


# -----------------------------
# Sidebar
# -----------------------------
if _has_logo():
    st.sidebar.image(str(LOGO_PATH), width=220)

st.sidebar.title("Agentic SQL RAG")
st.sidebar.caption(
    "A professional retrieval and grounding interface for indexed documents, evidence inspection, ranking analysis, and evaluation tracking."
)

api_base_url = st.sidebar.text_input("API Base URL", value=DEFAULT_API_BASE_URL)
api_base_url = _normalize_base_url(api_base_url)

ask_url = f"{api_base_url}/ask"
agent_ask_url = f"{api_base_url}/agent/ask"
documents_upload_url = f"{api_base_url}/documents/upload"
evaluation_summary_url = f"{api_base_url}/evaluation/summary"
evaluation_runs_url = f"{api_base_url}/evaluation/runs"

health, health_err = _get_health(api_base_url)
if health_err:
    st.sidebar.error(f"API unavailable: {health_err}")
    st.sidebar.caption(
        "Start FastAPI in another terminal with: "
        "python -m uvicorn app.main:app --env-file docker/.env"
    )
else:
    st.sidebar.success("API healthy")

st.sidebar.markdown("---")
st.sidebar.subheader("Document Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload a .txt or .md file",
    type=["txt", "md"],
    accept_multiple_files=False,
    key=f"document_uploader_{st.session_state['uploader_reset_counter']}",
)

col_up_1, col_up_2 = st.sidebar.columns(2)
upload_clicked = col_up_1.button("Upload & Index", use_container_width=True)
reset_clicked = col_up_2.button("Reset File", use_container_width=True)

if reset_clicked:
    st.session_state["last_upload_status"] = None
    st.session_state["uploader_reset_counter"] += 1
    st.rerun()

if uploaded_file is not None and upload_clicked:
    if health_err:
        st.session_state["last_upload_status"] = {
            "ok": False,
            "message": "API unavailable. Start FastAPI first.",
        }
    else:
        with st.sidebar:
            with st.spinner("Uploading and indexing document..."):
                upload_data, upload_err = _post_file(
                    documents_upload_url,
                    file_name=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                    mime_type=uploaded_file.type or "text/plain",
                )

        if upload_err:
            st.session_state["last_upload_status"] = {
                "ok": False,
                "message": upload_err,
            }
        else:
            doc_info = (
                upload_data.get("document", {}) if isinstance(upload_data, dict) else {}
            )
            st.session_state["last_upload_status"] = {
                "ok": True,
                "message": "Document uploaded and indexed.",
                "document_id": doc_info.get("document_id"),
                "chunks_inserted": doc_info.get("chunks_inserted"),
                "source": doc_info.get("source"),
                "title": doc_info.get("title"),
            }
            st.session_state["uploader_reset_counter"] += 1
            st.rerun()

upload_status = st.session_state.get("last_upload_status")
if upload_status:
    if upload_status.get("ok"):
        st.sidebar.success(upload_status.get("message", "Upload complete."))
        doc_id = upload_status.get("document_id")
        chunks_inserted = upload_status.get("chunks_inserted")
        source = upload_status.get("source")
        title = upload_status.get("title")

        details: list[str] = []
        if title:
            details.append(f"title={title}")
        if doc_id is not None:
            details.append(f"document_id={doc_id}")
        if chunks_inserted is not None:
            details.append(f"chunks={chunks_inserted}")
        if source:
            details.append(f"source={source}")

        if details:
            st.sidebar.caption(" · ".join(details))
    else:
        st.sidebar.error(upload_status.get("message", "Upload failed."))

st.sidebar.markdown("---")
mode_ui = st.sidebar.radio("Request Mode", ["Ask", "Agent Ask"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Retrieval Settings")
retrieval_mode = st.sidebar.selectbox("Mode", ["hybrid", "fts", "vector"], index=0)
alpha = st.sidebar.slider("Alpha", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
k_final = st.sidebar.slider("Top K Final", min_value=1, max_value=10, value=5, step=1)

st.sidebar.subheader("Reranking")
rerank = st.sidebar.checkbox("Enable Rerank", value=True)
rerank_method = st.sidebar.selectbox("Rerank Method", ["ml", "overlap"], index=0)
rerank_weight = st.sidebar.slider(
    "Rerank Weight", min_value=0.0, max_value=1.0, value=0.15, step=0.05
)
rerank_top_k = st.sidebar.slider(
    "Rerank Top K", min_value=1, max_value=10, value=5, step=1
)

st.sidebar.subheader("Answer Settings")
max_sentences = st.sidebar.slider(
    "Max Sentences", min_value=1, max_value=6, value=3, step=1
)
support_threshold = st.sidebar.slider(
    "Support Threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.05
)

show_raw_json = st.sidebar.checkbox("Show Raw JSON", value=False)

st.sidebar.markdown("---")
if st.sidebar.button("Clear chat", use_container_width=True):
    st.session_state["messages"] = []
    st.rerun()


# -----------------------------
# Main UI
# -----------------------------
header_col1, header_col2 = st.columns([2, 8])

with header_col1:
    if _has_logo():
        st.image(str(LOGO_PATH), width=180)
    else:
        st.markdown("## 🔎")

with header_col2:
    st.title("Agentic SQL RAG")
    st.write(
        "Ask grounded questions over indexed documents with citations, evidence cards, grounding validation, rerank summaries, document indexing, and evaluation insights."
    )

st.markdown("---")

# Evaluation dashboard
if not health_err:
    summary_data, summary_err = _fetch_evaluation_summary(evaluation_summary_url)
    runs_data, runs_err = _fetch_evaluation_runs(evaluation_runs_url)
else:
    summary_data, summary_err = None, "API unavailable"
    runs_data, runs_err = [], "API unavailable"

_render_evaluation_dashboard(summary_data, runs_data, summary_err, runs_err)

st.markdown("---")

# Existing chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg.get("content", ""))
        else:
            _render_assistant_message(msg, show_raw_json=show_raw_json)

user_query = st.chat_input("Ask a question about your indexed documents...")

if user_query:
    if not user_query.strip():
        st.error("Please enter a question.")
    elif health_err:
        st.error(
            "The API is currently unavailable. Start FastAPI first: "
            "python -m uvicorn app.main:app --env-file docker/.env"
        )
    else:
        _append_user_message(user_query)

        with st.chat_message("user"):
            st.markdown(user_query)

        if mode_ui == "Ask":
            payload = _build_ask_payload(
                query=user_query,
                retrieval_mode=retrieval_mode,
                alpha=alpha,
                k_final=k_final,
                rerank=rerank,
                rerank_method=rerank_method,
                rerank_weight=rerank_weight,
                rerank_top_k=rerank_top_k,
                max_sentences=max_sentences,
                support_threshold=support_threshold,
            )
            url = ask_url
        else:
            payload = _build_agent_payload(
                query=user_query,
                retrieval_mode=retrieval_mode,
                alpha=alpha,
                k_final=k_final,
            )
            url = agent_ask_url

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                data, err = _post_json(url, payload)

            if err:
                error_text = f"Backend error: {err}"
                st.error(error_text)
                _append_assistant_message(
                    content=error_text,
                    citations=[],
                    grounding={},
                    tools=[],
                    raw_json={"error": err},
                )
            elif not data:
                error_text = "Empty response from backend."
                st.error(error_text)
                _append_assistant_message(
                    content=error_text,
                    citations=[],
                    grounding={},
                    tools=[],
                    raw_json={},
                )
            else:
                answer = data.get("answer", "")
                citations = data.get("citations", [])
                grounding = data.get("grounding", {})
                tools = data.get("tools", []) if mode_ui == "Agent Ask" else []

                temp_msg = {
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "grounding": grounding,
                    "tools": tools,
                    "raw_json": data,
                }
                _render_assistant_message(temp_msg, show_raw_json=show_raw_json)
                _append_assistant_message(
                    content=answer,
                    citations=citations,
                    grounding=grounding,
                    tools=tools,
                    raw_json=data,
                )
