import streamlit as st
from app.static_rag import run_static_rag
from app.dynamic_rag import process_pdf
from app.website_rag import process_website
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

#EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en" )

st.set_page_config(page_title="AI Knowledge Assistant")
st.title(" AI Knowledge Assistant")
mode = st.radio("Choose Mode:",["Static Knowledge", "Upload PDF", "Website URL"] )

query = st.text_input("Ask a question")

# STATIC MODE
if mode == "Static Knowledge":
    if st.button("Ask"):
        answer = run_static_rag(query)
        st.write(answer)

# DYNAMIC MODE
if mode == "Upload PDF":
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        vectorstore = process_pdf(uploaded_file)

        if st.button("Ask"):
            results = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([r.page_content for r in results])

            llm = ChatOllama(model="llama3")

            prompt = f"""
            Answer using only the context below.

            Context:
            {context}

            Question:
            {query}
            """

            response = llm.invoke(prompt)
            st.write(response.content)

# WEBSITE MODE
if mode == "Website URL":
    url = st.text_input("Enter Website URL")

    if url:
        vectorstore = process_website(url)

        if st.button("Ask"):
            results = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([r.page_content for r in results])

            llm = ChatOllama(model="llama3")

            prompt = f"""
            Answer using only the context below.

            Context:
            {context}

            Question:
            {query}
            """

            response = llm.invoke(prompt)
            st.write(response.content)