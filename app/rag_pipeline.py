#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import load_documents, split_documents

EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en" )
PERSIST_DIRECTORY = "db"

def create_vector_store():
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    chunks = split_documents(documents)
    print(f"Total chunks: {len(chunks)}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDING_MODEL,
        persist_directory=PERSIST_DIRECTORY 
    )
    print("************** Vector store created successfully!**************")
    return vectorstore

def load_vector_store():
    vectorstore = Chroma( persist_directory=PERSIST_DIRECTORY,embedding_function=EMBEDDING_MODEL)
    return vectorstore
"""if __name__ == "__main__":
    print("Starting RAG pipeline...")
    create_vector_store()"""