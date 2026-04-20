from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

DATA_PATH = "data/documents"

def load_documents():
    documents = []
    for file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file)
        if file.endswith(".txt"):
            loader = TextLoader(file_path) # convert text in LangChain object
            docs = loader.load()
            documents.extend(docs)
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    return chunks

"""if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"Original documents: {len(docs)}")
    print(f"Total chunks: {len(chunks)}")

    print("\nSample chunk:\n")
    print(chunks[0].page_content)"""