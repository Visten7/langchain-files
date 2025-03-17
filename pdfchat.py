import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    st.set_page_config(page_title="ChatToPDF")
    st.header("ChatToPDF")

    # Upload PDF
    pdf = st.file_uploader("Upload a PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""

        # Extract text from PDF
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Load OpenAI API key securely
        openai_api_key = os.getenv("OPENAI_API_KEY")  # Get API key from environment variable
        if not openai_api_key:
            st.error("OpenAI API key is missing. Please set it in an environment variable or .env file.")
            return
        
        # Generate embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Ask a question
        user_question = st.text_input("Ask a question:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            # Load LLM
            llm = OpenAI(api_key=openai_api_key)

            # Load QA chain
            chain = load_qa_chain(llm, chain_type="stuff")

            # Get response from LLM
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)
        st.write("Knowledge base successfully created.")
# Run the application
if __name__ == "__main__":
    main()
