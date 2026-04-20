from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def process_website(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100 )
    chunks = splitter.split_documents(docs)

    EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en" )
    vectorstore = Chroma.from_documents(  documents=chunks,   embedding=EMBEDDING_MODEL    )
    return vectorstore