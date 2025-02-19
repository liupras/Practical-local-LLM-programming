#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-17
# @function: 用langgraph实现的有Query analysis的rag
# @version : V0.5

# https://python.langchain.com/docs/tutorials/rag/

import os 
os.environ['USER_AGENT'] = 'rag_graph_with_query_analysis'

"""
确定重要文件路径
"""

import sys

# 当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 当前文件所在的目录
current_dir = os.path.dirname(current_file_path)

# 待矢量化的源文件地址
src_url = os.path.join("http://wfcoding.com/articles/practice/0318/")

def get_persist_directory(model_name):
    """矢量数据库存储路径"""
    model_name = model_name.replace(":","-")
    return os.path.join(current_dir,f'assert/rag_{model_name}')

"""
1. 创建本地嵌入数据库
"""

embed_model_name = "shaw/dmeta-embedding-zh"
batch_size = 3

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def create_db(model_name,url):    
    """生成本地矢量数据库"""

    persist_directory = get_persist_directory(model_name)

    # 判断矢量数据库是否存在，如果存在则不再做索引，方便反复测试
    if os.path.exists(persist_directory):
        return
    
    embedding = OllamaEmbeddings(model=model_name)
    vectordb = Chroma(persist_directory=persist_directory,embedding_function=embedding)

    # 加载并分块博客内容
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-header","post-content")       # 指解析css class 为post-header和post-content 的内容
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # 在文档的 元数据(metadata) 中添加 section 标签
    
    total_documents = len(all_splits)
    third = total_documents // 3

    for i, document in enumerate(all_splits):
        if i < third:
            document.metadata["section"] = "开头"
        elif i < 2 * third:
            document.metadata["section"] = "中间"
        else:
            document.metadata["section"] = "结尾"

    print(f'Metadata: {all_splits[0].metadata}') 
    

    for i in tqdm(range(0, len(all_splits), batch_size), desc="嵌入进度"):
        batch = all_splits[i:i + batch_size]

        # 从文本块生成嵌入，并将嵌入存储在本地磁盘。
        vectordb.add_documents(batch)

 
create_db(embed_model_name,src_url)
vector_store = Chroma(persist_directory=get_persist_directory(embed_model_name),embedding_function=OllamaEmbeddings(model=embed_model_name))

def similarity_search_with_score(state):
    """矢量数据库检索测试
    返回文档评分，分数越高，文档越相似。
    """

    results = vector_store.similarity_search_with_score(
        state["query"],
        k = 2,
        filter={"section": state["section"]},
    )
    return results

"""
2. 检索和生成回答
"""

from typing_extensions import List, TypedDict, Annotated

class Search(TypedDict):
    """查询检索的参数"""

    query: Annotated[str, ..., "查询的关键词"]
    section: Annotated[str, ..., "要查询的部分，必须是'开头、中间、结尾'之一。"] 

class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

def retrieve(state: State):
    """检索"""

    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter={"section": query["section"]},
    )
    return {"context": retrieved_docs}

# 定义提示词
#prompt = hub.pull("rlm/rag-prompt")
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph

def create_graph(llm_model_name):
    """创建langgraph"""

    llm = ChatOllama(model=llm_model_name,temperature=0,verbose=True)

    def analyze_query(state: State):
        """分析查询，推理出查询参数"""
        
        structured_llm = llm.with_structured_output(Search)
        query = structured_llm.invoke(state["question"])
        return {"query": query}

    def generate(state: State):
        """生成回答"""

        prompt = ChatPromptTemplate.from_messages([
            ("human", """你是问答任务的助手。
            请使用以下检索到的**上下文**来回答问题。
            如果你不知道答案，就说你不知道。最多使用三句话，并保持答案简洁。
            
            问题: {question} 

            上下文: {context} 

            回答："""),
        ])
        
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
    graph_builder.add_edge(START, "analyze_query")
    graph = graph_builder.compile()

    return graph


def ask(llm_model_name,question):
    """问答"""

    graph = create_graph(llm_model_name)
    for step in graph.stream(
        {"question": question},
        stream_mode="updates",
    ):
        print(f"{step}\n\n----------------\n")


if __name__ == '__main__':

    similarity_search_with_score({"query":"langgraph 结尾","section":"结尾"})

    from utils import show_graph
    graph = create_graph("llama3.1")
    show_graph(graph)

    q = "文章的结尾讲了langgraph的哪些优点？"
    ask("qwen2.5",q)
    ask("llama3.1",q)
    ask("MFDoom/deepseek-r1-tool-calling:7b",q)
    


