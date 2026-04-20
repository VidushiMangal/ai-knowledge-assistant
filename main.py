import streamlit as st
from app.chat import ask_question

st.set_page_config(page_title="AI Knowledge Assistant")

st.title("AI Knowledge Assistant")
st.write("Ask questions from your developer notes.")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        with st.spinner("Thinking..."):
            answer = ask_question(query)
            st.subheader("Answer")
            st.write(answer)