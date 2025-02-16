#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-17
# @function: 用langgraph实现的rag
# @version : V0.5

# https://python.langchain.com/docs/tutorials/rag/

import os 
os.environ['USER_AGENT'] = 'rag_graph'

"""
确定文件路径
"""

import sys

# 当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 当前文件所在的目录
current_dir = os.path.dirname(current_file_path)

def get_persist_directory(model_name):
    """矢量数据库存储路径"""
    model_name = model_name.replace(":","-")
    return os.path.join(current_dir,f'assert/animals_{model_name}')

"""
1. 创建矢量数据库对象
"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embed_model_name = "shaw/dmeta-embedding-zh"
vector_store = Chroma(persist_directory=get_persist_directory(embed_model_name),embedding_function=OllamaEmbeddings(model=embed_model_name))

"""
2. 设置提示词
"""

from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

#prompt = hub.pull("rlm/rag-prompt")
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("human", """你是问答任务的助手。
     请使用以下检索到的**上下文**来回答问题。
     如果你不知道答案，就说你不知道。最多使用三句话，并保持答案简洁。
     
     问题: {question} 

     上下文: {context} 

     回答："""),
])

"""
3. 使用langgraph构建 RAG 系统
"""

class State(TypedDict):
    """状态：在 langgraph 中传递"""

    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    """节点：检索"""

    retrieved_docs = vector_store.similarity_search(state["question"],k=2)
    return {"context": retrieved_docs}

from langchain_ollama import ChatOllama

def build_graph(llm_model_name):
    """构建langgraph"""

    def generate(state: State):
        """节点：生成 """

        llm = ChatOllama(model=llm_model_name,temperature=0, verbose=True)
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    # 定义步骤
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph

def ask(llm_model_name,question):
    """提问"""

    print(f'--------{llm_model_name}----------')
    graph = build_graph(llm_model_name)
    response = graph.invoke({"question": question})
    print(f'the answer is: \n{response["answer"]}')

if __name__ == '__main__':

    graph = build_graph("qwen2.5")

    from utils import show_graph
    show_graph(graph)

    question = "大象的学名是什么？它有什么显著特点？对人类有什么帮助？"
    ask("qwen2.5",question)
    ask("deepseek-r1",question)
    ask("llama3.1",question)

