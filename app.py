from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_openai import OpenAI
from groq import Groq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain_groq import ChatGroq 
import pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')  # ✅ Add your Pinecone environment here
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "symptomsense1"

# ✅ Initialize Pinecone (old client style)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# ✅ Get the index
index = pinecone.Index(index_name)

# ✅ Use LangChain’s Pinecone wrapper
docsearch = LangchainPinecone(
    index=index,
    embedding=embeddings,
    text_key="text"   # make sure your docs use "text" field
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(model="llama3-8b-8192", temperature=0.4, max_tokens=500)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat2.html')


@app.route('/about')
def about():
    return render_template('about-us.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
