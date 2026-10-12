from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import os
import time
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

print("API key loaded:", bool(os.getenv("NVIDIA_API_KEY")))

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    temperature=0,
)
try:
    choice = int(input("Enter your choice: "))
except ValueError:
    choice = 1  # Default fallback if input isn't a number

# 2. Determine the system prompt using match-case
match choice:
    case 1:
        persona = "You are a funny AI agent that answers questions humorously. Give an emoji to every line in your answers."
    case 2:
        persona = "You are an angry AI agent that answers questions angrily. Give an angry emoji to every line in your answers."
    case 3:
        persona = "You are a sad AI agent that answers questions sadly. Give a sad emoji to every line in your answers."
    case _:
        # The wildcard (_) acts as the default 'else' case for invalid choices (e.g., 4, 99)
        persona = "You are a funny AI agent that answers questions humorously. Give an emoji to every line in your answers."

# 3. Create the messages array
messages = [
    SystemMessage(content=persona)
]
        

while True:

    prompt = input("\nEnter your prompt (or '0' to quit): ")

    if prompt == "0":
        break

    messages.append(HumanMessage(content=prompt))

    start = time.time()

    full_response = ""

    print("\n🤖 ", end="", flush=True)

    for chunk in llm.stream(messages):
        content = chunk.content

        if content:
            print(content, end="", flush=True)
            full_response += content

    messages.append(AIMessage(content=full_response))

    print(f"\n\n⏱️ Total time: {time.time() - start:.2f} seconds")

print(messages)