from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an intelligent Movie Analysis Agent.

Analyze the provided movie content and extract useful,
accurate and structured information.

Rules:
- Use only the provided movie content.
- Do not invent information.
- If information is unavailable, say "Not available".
- Focus on the user's request.
- Separate factual information from interpretation.
- Be concise but informative.
- Try to provide a structured response in bullet points or numbered lists under 150 words.
"""
    ),
    (
        "human",
        """Movie Content:
{movie_content}

User Request:
{user_query}"""
    ),
])

response = llm.invoke(
    prompt.invoke({
        "movie_content": """
        The movie follows a young engineer who returns to his
        hometown and discovers that his father disappeared years ago.
        He investigates the mystery and discovers a hidden conspiracy.
        """,
        "user_query": "What is the main conflict in the movie?"
    })
)

print(response.content)