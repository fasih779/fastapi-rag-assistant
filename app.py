import streamlit as st
import requests

API_URL = "http://localhost:8000/query"

st.set_page_config(
    page_title="FastAPI RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


st.markdown("""
<div class="hero">
    <h1>🤖 FastAPI RAG Assistant</h1>
    <p>Powered by Llama 3.3 • Qdrant • Sentence Transformers</p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask anything about the FastAPI codebase...")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):

        with st.spinner("Searching codebase..."):

            try:
                response = requests.post(
                    API_URL,
                    json={"question": prompt},
                    timeout=60
                )

                response.raise_for_status()
                answer = response.json()["answer"]

            except Exception as e:
                answer = str(e)

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )