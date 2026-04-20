import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

#from main import EMBEDDING_MODEL

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100   )
    chunks = splitter.split_documents(docs)

    EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en" )
    vectorstore = Chroma.from_documents(documents=chunks, embedding=EMBEDDING_MODEL   )
    return vectorstore