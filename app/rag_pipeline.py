#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import load_documents, split_documents

PERSIST_DIRECTORY = "db"

def create_vector_store():
    print("--------------------------------------------Step 1: Loading documents...")
    documents = load_documents()

    print(f"Loaded {len(documents)} documents")

    print("--------------------------------------------Step 2: Splitting documents...")
    chunks = split_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    print("--------------------------------------------Step 3: Creating embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )

    print("--------------------------------------------Step 4: Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY 
    )

    print("--------------------------------------------Step 5: Persisting database...")
    #vectorstore.persist() # to store data in local drive

    print("--------------------------------------------✅ Vector store created successfully!")

    return vectorstore


if __name__ == "__main__":
    print("Starting RAG pipeline...")
    create_vector_store()