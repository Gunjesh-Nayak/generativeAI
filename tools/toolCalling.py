from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from rich import print

@tool#tool creation
def get_text_length(text: str) -> int:
    """Returns the length of the given text."""
    return len(text)

llm = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    temperature=0.5,
)
#tool binding
llm_tool = llm.bind_tools([get_text_length])

# result=llm.invoke("Returns the length of the given text: Hello, world!")
result1=llm_tool.invoke("Returns the length of the given text: Hello, world!")
# print(result)
# print()
# print()
# print()
# print()
# print()

# print(result1.tool_calls[0])

if(result1.tool_calls):
    tool_call = result1.tool_calls[0]

    print(f"Tool Name: {tool_call['name']}")
    # print(f"Tool Description: {tool_call['description']}")    
    tool_name=tool_call['name']
    # tool_description=tool_call['description']
    tool_args=tool_call['args']

tool_result=get_text_length.invoke(tool_args)
final_response=llm_tool.invoke(f"Returns the length of the given text: {tool_result}")
print(final_response)
print(final_response.content)
    






























# from dotenv import load_dotenv
# load_dotenv()

# from langchain.tools import tool
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain_core.messages import ToolMessage
# from rich import print


# @tool
# def get_text_length(text: str) -> int:
#     """Returns the length of the given text."""
#     return len(text)


# llm = ChatNVIDIA(
#     model="openai/gpt-oss-20b",
#     temperature=0.5,
# )

# # Bind tool to LLM
# llm_tool = llm.bind_tools([get_text_length])


# # 1. Ask LLM
# result1 = llm_tool.invoke(
#     "What is the length of this text: Hello, world?"
# )

# print("FIRST LLM RESPONSE:")
# print(result1)

# print("\nTOOL CALL:")
# print(result1.tool_calls)


# # 2. Check whether LLM requested a tool
# if result1.tool_calls:

#     tool_call = result1.tool_calls[0]

#     print("\nTool Name:", tool_call["name"])
#     print("Tool Arguments:", tool_call["args"])
#     print("Tool Call ID:", tool_call["id"])

#     # 3. Execute the tool
#     tool_result = get_text_length.invoke(tool_call["args"])

#     print("\nTOOL RESULT:")
#     print(tool_result)

#     # 4. Give result back to LLM
#     tool_message = ToolMessage(
#         content=str(tool_result),
#         tool_call_id=tool_call["id"]
#     )

#     final_response = llm_tool.invoke([
#         result1,
#         tool_message
#     ])

#     print("\nFINAL RESPONSE:")
#     print(final_response.content)

# else:
#     print("\nLLM did not request a tool.")
#     print(result1.content)