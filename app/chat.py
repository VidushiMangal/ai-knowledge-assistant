from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from app.rag_pipeline import load_vector_store
PERSIST_DIRECTORY = "db"


def ask_question(query):
    vectorstore = load_vector_store()
    # Retrieve relevant chunks
    results = vectorstore.similarity_search(query, k=2)
    context = "\n\n".join([r.page_content for r in results])
    # Connect to LLM
    llm = ChatOllama(model="llama3")
    prompt = f"""
    You are an AI Developer Knowledge Assistant.
    Answer ONLY using the provided context.

    Rules:
    1. If answer exists in context, explain clearly.
    2. If answer not found, say: "I could not find that in the provided documents."
    3. Keep answer concise and structured.
    4. Use bullet points when useful.

    Context:
    {context}

    Question:
    {query}

Answer:
"""

    response = llm.invoke(prompt)
    print("\nAnswer:\n")
    return response.content

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        ask_question(query)