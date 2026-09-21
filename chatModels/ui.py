import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Funny NVIDIA AI",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# Basic styling
# -----------------------------
st.markdown("""
<style>
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
    }

    .chat-title {
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .chat-subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }

    .status {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<h1 class="chat-title">🤖 Funny NVIDIA AI</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="chat-subtitle">Powered by Nemotron 3.5 Lightning 30B A3B</p>',
    unsafe_allow_html=True
)

# -----------------------------
# NVIDIA model
# -----------------------------
llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    temperature=0,
)

# -----------------------------
# Conversation state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content=(
                "You are a funny AI agent that answers questions humorously. "
                "Give an emoji to every line in your answers."
            )
        )
    ]

# -----------------------------
# Display previous messages
# -----------------------------
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)

# -----------------------------
# Chat input
# -----------------------------
prompt = st.chat_input("Ask me anything...")

if prompt:
    # Add and display user message
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream NVIDIA response
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in llm.stream(st.session_state.messages):
                content = chunk.content

                if content:
                    full_response += content
                    response_placeholder.markdown(full_response)

            # Keep one complete AI message in history
            st.session_state.messages.append(
                AIMessage(content=full_response)
            )

        except Exception as e:
            response_placeholder.error(f"Error: {e}")
            st.session_state.messages.pop()

st.markdown(
    '<div class="status">NVIDIA Nemotron • Streaming responses</div>',
    unsafe_allow_html=True
)
