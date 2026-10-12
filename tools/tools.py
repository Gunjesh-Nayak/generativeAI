from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(max_results=3)
llm = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    temperature=0.5,
)
prompt= ChatPromptTemplate.from_template("""
You are a helpful assistant that extracts relevant information from the search results into bullets points.
{news}
""")

#runnable
# chain= prompt | llm | StrOutputParser()

# result=search_tool.run("What are the latest news about generative AI 2026,13 september?")

# news=chain.invoke({"news": result})
# print(news)

print(search_tool.name)
print(search_tool.description)
print(search_tool.args)

# tools make
#tools bind
#tools call
#tools execute
#