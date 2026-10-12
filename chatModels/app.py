
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
    page_title="NVIDIA AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# UI Styling
# -----------------------------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }

    .role-box {
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">🤖 NVIDIA AI Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Choose an AI personality and start chatting</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Role Selection
# -----------------------------
choice = st.selectbox(
    "Choose AI Personality",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "😂 Funny AI",
        2: "😡 Angry AI",
        3: "😢 Sad AI"
    }[x]
)

# -----------------------------
# Persona
# -----------------------------
match choice:
    case 1:
        persona = (
            "You are a funny AI agent that answers questions humorously. "
            "Give an emoji to every line in your answers."
        )
        avatar = "😂"

    case 2:
        persona = (
            "You are an angry AI agent that answers questions angrily. "
            "Give an angry emoji to every line in your answers."
        )
        avatar = "😡"

    case 3:
        persona = (
            "You are a sad AI agent that answers questions sadly. "
            "Give a sad emoji to every line in your answers."
        )
        avatar = "😢"

# -----------------------------
# NVIDIA Model
# -----------------------------
llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    temperature=0,
)

# -----------------------------
# Initialize conversation
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=persona)
    ]

# If user changes personality, update system message
if st.session_state.get("choice") != choice:

    st.session_state.choice = choice

    # Keep only the system message when changing personality
    st.session_state.messages = [
        SystemMessage(content=persona)
    ]

# -----------------------------
# Display chat history
# -----------------------------
for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant", avatar=avatar):
            st.markdown(message.content)

# -----------------------------
# Chat Input
# -----------------------------
prompt = st.chat_input("Enter your prompt...")

if prompt:

    # Add user message
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant", avatar=avatar):

        response_placeholder = st.empty()
        full_response = ""

        try:

            for chunk in llm.stream(st.session_state.messages):

                content = chunk.content

                if content:

                    full_response += content

                    response_placeholder.markdown(
                        full_response
                    )

            # Store ONE complete AI message
            st.session_state.messages.append(
                AIMessage(content=full_response)
            )

        except Exception as e:

            response_placeholder.error(
                f"Error: {e}"
            )

            # Remove user message if request fails
            st.session_state.messages.pop()

