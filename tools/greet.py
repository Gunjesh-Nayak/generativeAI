from langchain.tools import tool


@tool
def greet(name: str) -> str: #typeHints
    """Generates a greeting message for the given name.""" #docString
    return f"Hello, {name}!"

# print(greet("gunjesh"))
result=greet.invoke({"name": "gunjesh"})
print(result)
print(greet.name)
print(greet.description)
print(greet.args)