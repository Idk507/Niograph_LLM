from langchain.document_loaders import TextLoader
import textwrap
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
import os
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YourHuggingFaceAPI"
DB_FAISS_PATH = 'vectorstore_faiss/db'
loader = TextLoader("./data/nio_complete.txt")
document = loader.load()
print(document)

def wrap(text):
    lines = text.split("\n")
    wrapped_lines = [textwrap.fill(line,width = 110) for line in lines]
    wrapped_text = "\n".join(wrapped_lines)
    return wrapped_text

wrap(str(document[0]))

text_splitter = CharacterTextSplitter(chunk_size = 500,chunk_overlap = 50)
docs = text_splitter.split_documents(document)

embeddings = HuggingFaceBgeEmbeddings()

db = FAISS.from_documents(docs,embeddings)

query = "Demo"
doc = db.similarity_search(query)
print(wrap(str(doc[0])))

llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl",model_kwargs = {"temperature":0.8,"max_length":100})
chain = load_qa_chain(llm,chain_type = "stuff")

query_text ="Is niograph a big company"
doc_result = db.similarity_search(query_text)
print(chain.run(input_documents = doc_result,question = query_text))

def call(query):
    if query == "":
        print("Please Enter text")
    else:
        doc_result = db.similarity_search(query_text)
        return chain.run(input_documents = doc_result,question = query_text)
    
    

