#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-17
# @function: 用langgraph实现的有Query analysis的rag
# @version : V0.5

# https://python.langchain.com/docs/tutorials/rag/

import os 
os.environ['USER_AGENT'] = 'rag_graph_with_query_analysis'

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

# Query analysis
total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"


print(f'Metadata: {all_splits[0].metadata}') 

# Index chunks
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(documents=all_splits)

from typing import Literal

from typing_extensions import Annotated


class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[str, ..., "Section to query,Must be one of beginning, middle, end."] 

class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}


def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}

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


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()


def ask(question):
    for step in graph.stream(
        {"question": question},
        stream_mode="updates",
    ):
        print(f"{step}\n\n----------------\n")


def show_graph():
    from PIL import Image as PILImage
    from io import BytesIO
    png_data = graph.get_graph().draw_mermaid_png()
    img = PILImage.open(BytesIO(png_data))
    img.show()


if __name__ == '__main__':
    ask("What does the beginning of the post say about Task Decomposition?")
    #show_graph()