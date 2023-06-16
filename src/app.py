from fastapi import FastAPI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import ElasticVectorSearch
import openai
from langchain.llms import AzureOpenAI

openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = "https://cog-vlnygtsnpw4pe.openai.azure.com/"
openai.api_key = "02964635d4a1475c80a6c326736df0b8"

embedding = OpenAIEmbeddings(deployment="embedding", model="text-embedding-ada-002", chunk_size=1)
print(embedding)

db = ElasticVectorSearch(
    elasticsearch_url="https://elastic:=OAxbvmXbY0oRydGrYWG@elasticsearch.itxbpm.com:9200/",
    index_name="chat",
    embedding=embedding,
)
qa = RetrievalQA.from_chain_type(
    llm = AzureOpenAI(deployment_name="davinci"),
    chain_type="stuff",
    retriever=db.as_retriever(),
)

app = FastAPI()


@app.get("/")
def index():
    return {
        "message": "Make a post request to /ask to ask questions about Meditations by Marcus Aurelius"
    }


@app.post("/ask")
def ask(query: str):
    response = qa.run(query)
    return {
        "response": response,
    }
