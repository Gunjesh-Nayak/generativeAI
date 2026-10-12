from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()
llm = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    temperature=0,
    #  extra_body={
    #     "chat_template_kwargs": {
    #         "enable_thinking": False
    #     }
    # }

)

class Movie(BaseModel):
    title: str
    release_date: Optional[int]
    genere: list[str]
    director: Optional[str]
    cast: Optional[List[str]]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
     Extract movie information from the provided content and return it in a structured format as given to store in database no text only JSON: {formated_instructions}
    """),
    ("human", "{para}"),
])

response=llm.invoke(prompt.invoke({"para": "3 Idiots is a 2009 Indian coming-of-age comedy-drama directed by Rajkumar Hirani. The film stars Aamir Khan, R. Madhavan, Sharman Joshi, Kareena Kapoor, and Boman Irani. Set in an engineering college, the story explores themes of friendship, academic pressure, and following one's passion rather than societal expectations. The movie became one of the highest-grossing Indian films of its time and received immense praise for its humor and emotional depth. It has a rating of 8.4 on IMDb and remains a cultural favorite across generations.", 
"formated_instructions": parser.get_format_instructions()}))
print(response.content)
