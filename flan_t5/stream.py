import streamlit as st
from langchain.document_loaders import TextLoader
import textwrap
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
import os
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

# Set Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
DB_FAISS_PATH = 'vectorstore_faiss/db'

# Load document
loader = TextLoader("./data/nio_complete.txt")
document = loader.load()

# Wrap text
def wrap(text):
    lines = text.split("\n")
    wrapped_lines = [textwrap.fill(line, width=110) for line in lines]
    wrapped_text = "\n".join(wrapped_lines)
    return wrapped_text

# Split document into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(document)

# Create embeddings and FAISS database
embeddings = HuggingFaceBgeEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# Load QA chain model
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.8, "max_length": 100})
chain = load_qa_chain(llm, chain_type="stuff")

# Define Streamlit app
def main():
    st.title("Chatbot For Niograph")

    # User input
    user_input = st.text_input("Ask me anything:")

    if st.button("Ask"):
        if user_input == "":
            st.warning("Please enter a question.")
        else:
            # Get chatbot response
            response = call(user_input)
            st.text_area("Chatbot Response:", value=response, height=200)

# Chatbot function
def call(query):
    doc_result = db.similarity_search(query)
    return chain.run(input_documents=doc_result, question=query)

# Run Streamlit app
if __name__ == "__main__":
    main()
