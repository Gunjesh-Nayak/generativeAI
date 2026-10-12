from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_nvidia import ChatNVIDIA
from langchain_classic.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from rich import print
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from pathlib import Path
from sentence_transformers import CrossEncoder

load_dotenv()

llm=ChatNVIDIA(model="openai/gpt-oss-20b")#,max_completion_tokens=540)

pdf=PyMuPDFLoader('RAG/deep.pdf')

docs=pdf.load()
# print(docs[50].page_content)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(docs)

# print(texts)

embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
chroma_db = Path("./chroma_db")


if chroma_db.exists():
    print("Loading existing Chroma DB...")

    vectorstore = Chroma(
        persist_directory=str(chroma_db),
        embedding_function=embedding_model
    )

else:
    print("Creating Chroma DB...")

    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embedding_model,
        persist_directory=str(chroma_db)
    )

# print(vectorstore)



# Vector retriever
vector_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 10,
        # "fetch_k": 10
    }
)

# Keyword retriever
bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 5

# Hybrid retriever==============IMPORTANT+====================
retriever = EnsembleRetriever(
    retrievers=[
        vector_retriever,
        bm25_retriever
    ],
    weights=[
        0.5,
        0.5
    ]
)


SYSTEM_PROMPT = """
You are a teacher answering questions from a retrieved document.

Use ONLY information explicitly present in the Context.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- Keep the answer concise.
- Answer only what the user asked.
- Prefer 3-5 sentences unless the context requires more.
- If the answer is not in the context, say:
  "The provided context does not contain enough information to answer this question."

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

query=input("input : ")
# query = "Auxiliary Learning"

candidates = retriever.invoke(query)

context_text = "\n\n".join(
    doc.page_content for doc in candidates
)




#reranker++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
pairs = [
    [query, doc.page_content]
    for doc in candidates
]

scores = reranker.predict(pairs)

ranked_results = sorted(
    zip(scores, candidates),
    key=lambda x: x[0],
    reverse=True
)

final_prompt=prompt.invoke({
    "context":context_text,
    "question":query
})

context = [
    doc
    for score, doc in ranked_results[:3]
]

for rank, (score, doc) in enumerate(ranked_results[:3], 1):
    print(f"\n===== RANK {rank} =====")
    print("Score:", score)
    print("Page:", doc.metadata.get("page"))
    print(doc.page_content[:500])
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("\n=====================LLM====================\n")
import time

start = time.perf_counter()
first_token = None
output = ""

for chunk in llm.stream(
    final_prompt,
    reasoning_effort="low",
):
    if first_token is None:
        first_token = time.perf_counter()
        print(chunk)

    output += chunk.content

print(output)

end = time.perf_counter()

ttft = first_token - start
total = end - start

print(f"TTFT: {ttft:.2f}s")
print(f"Total: {total:.2f}s")
print(f"Generation after TTFT: {total - ttft:.2f}s")