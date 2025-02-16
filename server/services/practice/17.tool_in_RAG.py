#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-11
# @function: 使用tool的简单RAG
# @version : V0.5
# @Description ：去掉answer_style后回答比较好，有这个参数时回答很离谱。

# https://python.langchain.com/docs/how_to/convert_runnable_to_tool/

"""
确定文件路径
"""

import os,sys

# 将上级目录加入path，这样就可以引用上级目录的模块不会报错
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# 当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 当前文件所在的目录
current_dir = os.path.dirname(current_file_path)

def get_persist_directory(model_name):
    """矢量数据库存储路径"""
    model_name = model_name.replace(":","-")
    return os.path.join(current_dir,f'assert/animals_{model_name}')

"""
1. 创建检索器
"""
from common.MyVectorDB import LocalVectorDBChroma
def create_retriever(embed_model_name):
    """创建检索器"""

    persist_directory = get_persist_directory(embed_model_name)
    db = LocalVectorDBChroma(embed_model_name,persist_directory)

    # 基于Chroma 的 vector store 生成 检索器
    vector_store = db.get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )
    return retriever

embed_model_name = "shaw/dmeta-embedding-zh"
retriever = create_retriever(embed_model_name)

def search(query):
    """查询矢量数据库"""

    persist_directory = get_persist_directory(embed_model_name)
    db = LocalVectorDBChroma(embed_model_name,persist_directory)
    vector_store = db.get_vector_store()

    results = vector_store.similarity_search_with_score(query,k=1)
    return results

"""
2. 设置提示词
"""

from langchain_ollama import ChatOllama
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

system_prompt = """
您是问答任务的助手。
请使用以下**上下文**回答问题。如果您不知道答案，就说您不知道。
最多使用三句话并保持答案简洁。

下面请回答问题。

问题: {question}

上下文: {context}
"""

prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])

"""
3. RAG 链
"""

def create_rag_chain(llm_model_name):
    """创建RAG链"""

    llm = ChatOllama(model=llm_model_name,temperature=0,verbose=True)
    rag_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f'json_schema:{rag_chain.input_schema.model_json_schema()}')
    
    return rag_chain


def test_rag_chain(llm_model_name,question):
    """测试 rag 链"""

    print(f"------------{llm_model_name}-----------")
    rag_chain = create_rag_chain(llm_model_name)

    res = rag_chain.invoke({"question":question})
    print(res)

def test_rag_chain_stream(llm_model_name,question):
    """测试 rag 链，流式输出"""

    print(f"------------{llm_model_name}-----------")
    rag_chain = create_rag_chain(llm_model_name)
    for chunk in rag_chain.stream({"question":question}):
        print(chunk,end="-")

"""
4. 智能体
"""
from langgraph.prebuilt import create_react_agent

def create_agent(llm_model_name):
    """生成智能体"""

    rag_chain = create_rag_chain(llm_model_name)
    rag_tool = rag_chain.as_tool(
        name="animal_expert",
        description="获取有关动物的信息。",
    )

    llm = ChatOllama(model=llm_model_name,temperature=0,verbose=True)
    agent = create_react_agent(llm, [rag_tool])

    return agent

def test_agent(llm_model_name,question):

    agent = create_agent(llm_model_name)

    for chunk in agent.stream(
        {"messages": [("human",question)]}
    ):
        print(chunk)
        print("----")

if __name__ == '__main__':

    query = "猪的学名是什么？它对人类有什么用处？"
    '''
    test_rag_chain("qwen2.5",query)
    test_rag_chain_stream("qwen2.5",query)

    test_rag_chain("deepseek-r1",query)
    test_rag_chain_stream("deepseek-r1",query)
    
    test_rag_chain("llama3.1",query)
    test_rag_chain_stream("llama3.1",query)
    '''

    query = "蜜蜂的特点是什么？它对人类社会有什么作用？"
    test_agent("qwen2.5",query)
    test_agent("llama3.1",query)
    #test_agent("deepseek-r1",query)        # 不支持stream
