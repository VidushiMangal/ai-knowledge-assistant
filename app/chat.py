from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

PERSIST_DIRECTORY = "db"

def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    vectorstore = Chroma( persist_directory=PERSIST_DIRECTORY,embedding_function=embedding_model)
    return vectorstore

def ask_question(query):
    vectorstore = load_vector_store()
    # Retrieve relevant chunks
    results = vectorstore.similarity_search(query, k=2)
    context = "\n\n".join([r.page_content for r in results])
    # Connect to LLM
    llm = ChatOllama(model="llama3")
    prompt = f"""
    You are a helpful assistant.
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    print("\nAnswer:\n")
    print(response.content)


if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        ask_question(query)