#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-11
# @function: 在Agent中使用tool
# @version : V0.5
# @Description ：

# https://python.langchain.com/docs/how_to/convert_runnable_to_tool/

"""
1.确定重要文件路径
"""

import os,sys

# 将上级目录加入path，这样就可以引用上级目录的模块不会报错
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# 当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 当前文件所在的目录
current_dir = os.path.dirname(current_file_path)

# csv源文件地址
src_file_path = os.path.join(current_dir,'assert/animals.csv')

def get_persist_directory(model_name):
    """矢量数据库存储路径"""
    model_name = model_name.replace(":","-")
    return os.path.join(current_dir,f'assert/animals_{model_name}')

"""
2.在本地生成嵌入数据库
"""

from common.MyVectorDB import LocalVectorDBChroma
def create_db(model_name):    
    """生成本地矢量数据库"""

    persist_directory = get_persist_directory(model_name)
    if os.path.exists(persist_directory):
        return

    db = LocalVectorDBChroma(model_name,persist_directory)    
    db.embed_csv(src_file_path)

"""
3.智能体
"""

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

def ask_agent(embed_model_name,chat_modal_name,query):
    """测试智能体"""

    persist_directory = get_persist_directory(embed_model_name)
    db = LocalVectorDBChroma(embed_model_name,persist_directory)

    # 基于Chroma 的 vector store 生成 检索器
    vector_store = db.get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    # 将 检索器 包装为 工具
    tools = [
        retriever.as_tool(
            name="animal_info_retriever",
            description="查询动物的信息",
        )
    ]

    llm = ChatOllama(model=chat_modal_name,temperature=0.1,verbose=True)
    agent = create_react_agent(llm, tools)

    # 显示智能体的详细内容
    for chunk in agent.stream({"messages": [("human", query)]}):
        print(chunk)
        print("----")

def test_model(embed_model_name,chat_modal_name):
    print(f'\n---------------------{embed_model_name}-----------------------------')
    create_db(embed_model_name)

    query = "猪的学名是什么？它对人类有什么用处？"
    ask_agent(embed_model_name,chat_modal_name,query)

    query = "蜜蜂的特点是什么？它对人类社会有什么作用？"
    ask_agent(embed_model_name,chat_modal_name,query)

if __name__ == '__main__':

    test_model("shaw/dmeta-embedding-zh","qwen2.5")
    test_model("milkey/m3e","qwen2.5")
    test_model("mxbai-embed-large","qwen2.5")

    test_model("nomic-embed-text","llama3.1")
    test_model("all-minilm:33m","llama3.1")

    test_model("llama3.1","llama3.1")
    test_model("qwen2.5","qwen2.5")
