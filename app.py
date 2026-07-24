import streamlit as st
import requests

API_URL = "http://localhost:8000/query"
HEALTH_URL = "http://localhost:8000/health"

st.set_page_config(
    page_title="Codebase RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    collection = st.text_input(
        "Qdrant Collection",
        value="fastapi_codebase",
        help="Name of the Qdrant collection you indexed with ingest.py"
    )

    st.markdown("---")
    st.markdown("### 📚 Index a New Codebase")
    st.code(
        "python ingest.py \\\n"
        "  --path ./your_project \\\n"
        "  --collection your_project",
        language="bash"
    )

    st.markdown("---")
    # Health check
    try:
        health = requests.get(HEALTH_URL, timeout=3).json()
        loaded = health.get("loaded_collections", [])
        st.success(f"✅ API online")
        if loaded:
            st.markdown(f"**Loaded collections:** `{'`, `'.join(loaded)}`")
    except Exception:
        st.error("❌ API offline — start with:\n`uvicorn api:app --port 8000`")

    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding: 1.5rem 0 0.5rem 0;">
    <h1 style="margin:0;">🤖 Codebase RAG Assistant</h1>
    <p style="color:#888; margin:0.25rem 0 0 0;">
        Powered by Llama 3.3 &bull; Qdrant &bull; Sentence Transformers &bull;
        querying <code>{collection}</code>
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Chat history ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
prompt = st.chat_input(f"Ask anything about '{collection}'...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching codebase..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": prompt, "collection": collection},
                    timeout=60
                )
                response.raise_for_status()
                answer = response.json()["answer"]
            except Exception as e:
                answer = f"❌ Error: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})