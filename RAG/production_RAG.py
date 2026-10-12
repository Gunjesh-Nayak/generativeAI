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
import numpy as np

load_dotenv()

class SemanticCache:
    def __init__(self, embedding_model, threshold=0.85):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.cache = []

    def similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)

        return np.dot(a, b) / (
            np.linalg.norm(a) * np.linalg.norm(b)
        )

    def get(self, query):
        if not self.cache:
            return None

        query_embedding = self.embedding_model.embed_query(query)

        best_score = -1
        best_answer = None

        for item in self.cache:

            score = self.similarity(
                query_embedding,
                item["embedding"]
            )

            if score > best_score:
                best_score = score
                best_answer = item["answer"]

        print(f"\nCache similarity: {best_score:.3f}")

        if best_score >= self.threshold:
            print("CACHE HIT")
            return best_answer

        print("CACHE MISS")
        return None

    def set(self, query, answer):

        query_embedding = self.embedding_model.embed_query(query)

        self.cache.append({
            "query": query,
            "embedding": query_embedding,
            "answer": answer
        })

llm=ChatNVIDIA(model="openai/gpt-oss-20b",max_completion_tokens=540)

pdf=PyMuPDFLoader('RAG/deep.pdf')

docs=pdf.load()
# print(docs[50].page_content)

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# texts = text_splitter.split_documents(docs)
# print(texts)
#=============================================================
#                   Parent-Child Text Splitter
#=============================================================
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

parents = parent_splitter.split_documents(docs)

parent_lookup = {
    i: parent
    for i, parent in enumerate(parents)
}

child_docs = []

for parent_id, parent in enumerate(parents):

    children = child_splitter.split_documents([parent])

    for child in children:
        child.metadata["parent_id"] = parent_id
        child_docs.append(child)


embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

semantic_cache = SemanticCache(
    embedding_model,
    threshold=0.85
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
        documents=child_docs,
        embedding=embedding_model,
        persist_directory=str(chroma_db)
    )

# print(vectorstore)



# Vector retriever
vector_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 10,
        # "fetch_k": 30
    }
)

# Keyword retriever
bm25_retriever = BM25Retriever.from_documents(child_docs)
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
- Prefer 5-8 sentences unless the context requires more.
- If the answer is not in the context, say:
  "The provided context does not contain enough information to answer this question."

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# nli_model = CrossEncoder(
#     "cross-encoder/nli-deberta-v3-small"
# )

# import re
# def check_faithfulness(context, answer, threshold=0.60):  
#                                                #not usefull non factual answer/ book ,etc.

#     # Split answer into sentences
#     claims = [
#         claim.strip()
#         for claim in re.split(r'(?<=[.!?])\s+', answer)
#         if claim.strip()
#     ]

#     faithful_claims = 0

#     print("\n========== NLI FAITHFULNESS ==========")

#     for i, claim in enumerate(claims, 1):

#         logits = nli_model.predict(
#             [[context, claim]]
#         )[0]

#         # logits -> probabilities
#         exp_logits = np.exp(
#             logits - np.max(logits)
#         )

#         probabilities = (
#             exp_logits / exp_logits.sum()
#         )

#         labels = nli_model.model.config.id2label

#         scores = {
#             labels[j].lower(): float(probabilities[j])
#             for j in range(len(probabilities))
#         }

#         entailment = scores["entailment"]

#         print(f"\nClaim {i}: {claim}")
#         print(f"  Contradiction: {scores['contradiction']:.3f}")
#         print(f"  Entailment:    {entailment:.3f}")
#         print(f"  Neutral:       {scores['neutral']:.3f}")

#         if (
#             entailment >= threshold
#             and entailment > scores["contradiction"]
#             and entailment > scores["neutral"]
#         ):
#             print("  ✅ Supported")
#             faithful_claims += 1
#         else:
#             print("  ❌ Unsupported")

#     if not claims:
#         return False

#     faithfulness = faithful_claims / len(claims)

#     print(
#         f"\nFaithfulness: "
#         f"{faithful_claims}/{len(claims)} "
#         f"= {faithfulness:.2%}"
#     )

#     # Require at least 70% of claims to be supported
#     return faithfulness >= 0.70

while True:

    query = input("input : ").strip()

    if query == "0":
        break

    # ==========================================
    # SEMANTIC CACHE
    # ==========================================

    cached_answer = semantic_cache.get(query)

    if cached_answer is not None:

        print("\n===================== CACHE =====================")
        print(cached_answer)

        # IMPORTANT:
        # Skip retrieval + reranker + LLM
        continue


    # ==========================================
    # HYBRID RETRIEVAL
    # ==========================================

    candidates = retriever.invoke(query)
    parent_ids = []

    for doc in candidates:

        parent_id = doc.metadata["parent_id"]

        if parent_id not in parent_ids:
            parent_ids.append(parent_id)

    parent_results = [
    parent_lookup[parent_id]
    for parent_id in parent_ids
    ]

    # ==========================================
    # RERANKER
    # ==========================================

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

    # ==========================================
    # RELEVANCE THRESHOLD GATE
    # ==========================================

    RELEVANCE_THRESHOLD = 4.0

    best_score = ranked_results[0][0]

    print(f"\nBest relevance score: {best_score:.4f}")
    print(f"Relevance threshold: {RELEVANCE_THRESHOLD:.4f}")

    if best_score < RELEVANCE_THRESHOLD:

        print("\n❌ RELEVANCE GATE: FAILED")

        output = (
            "The provided context does not contain enough "
            "information to answer this question."
        )

        print("\n===================== LLM =====================\n")
        print(output)

        # Do NOT cache this answer
        continue

    print("\n✅ RELEVANCE GATE: PASSED")
    # ==========================================
    # TOP 3
    # ==========================================

    top_children = [
    doc
    for score, doc in ranked_results[:3]
    ]

    parent_ids = []

    for child in top_children:

        parent_id = child.metadata["parent_id"]

        if parent_id not in parent_ids:
            parent_ids.append(parent_id)

    context = [
        parent_lookup[parent_id]
        for parent_id in parent_ids
    ]

    context_text = "\n\n".join(
        doc.page_content
        for doc in context
    )
    

    

    # ==========================================
    # PROMPT
    # ==========================================

    final_prompt = prompt.invoke({
        "context": context_text,
        "question": query
    })


    # ==========================================
    # DEBUG RETRIEVAL
    # ==========================================

    for rank, (score, doc) in enumerate(
        ranked_results[:3], 1
    ):

        print(f"\n===== RANK {rank} =====")
        print(f"Score: {score:.4f}")
        print("Page:", doc.metadata.get("page"))
        print(doc.page_content[:500])


    # ==========================================
    # LLM
    # ==========================================

    print("\n===================== LLM =====================\n")

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

        print(
            chunk.content,
            end="",
            flush=True
        )

        output += chunk.content

    print()

    # ==========================================
    # NLI FAITHFULNESS CHECK
    # ==========================================

    # faithful = check_faithfulness(
    # context_text,
    # output
    # )

    # if not faithful:
    #     print("\n⚠️ Answer failed faithfulness check.")
    #     continue

    # semantic_cache.set(query, output)

    # ==========================================
    # LATENCY
    # ==========================================

    end = time.perf_counter()

    ttft = first_token - start
    total = end - start

    print(f"\nTTFT: {ttft:.2f}s")
    print(f"Total: {total:.2f}s")
    print(
        f"Generation after TTFT: "
        f"{total - ttft:.2f}s"
    )


    # ==========================================
    # STORE ANSWER IN CACHE
    # ==========================================

    semantic_cache.set(
        query,
        output
    )

    print("\n[Answer stored in semantic cache]")