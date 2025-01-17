#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-17
# @function: 用langgraph实现的rag
# @version : V0.5

# https://python.langchain.com/docs/tutorials/rag/

import os 
os.environ['USER_AGENT'] = 'rag_graph'

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1",temperature=0.3,verbose=True)

from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

import bs4
#from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
#prompt = hub.pull("rlm/rag-prompt")
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("human", """You are an assistant for question-answering tasks. 
     Use the following pieces of retrieved context to answer the question. 
     If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
     
     Question: {question} 

     Context: {context} 

     Answer:"""),
])

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

def ask(question):
    response = graph.invoke({"question": question})
    print(f'the answer is: \n{response["answer"]}')

def show_graph():
    from PIL import Image as PILImage
    from io import BytesIO
    png_data = graph.get_graph().draw_mermaid_png()
    img = PILImage.open(BytesIO(png_data))
    img.show()


if __name__ == '__main__':
    #ask("What is Task Decomposition?")
    show_graph()