#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-19
# @function: sql问答agent
# @version : V0.5
# @Description ：sql问答agent。

# https://python.langchain.com/v0.2/docs/tutorials/sql_qa/


"""
确定文件位置
"""
import os

# 获取当前执行的程序文件的文件夹路径
current_folder = os.path.dirname(os.path.abspath(__file__))

db_file_path = os.path.join(current_folder, 'assert/Chinook.db')


from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri(f"sqlite:///{db_file_path}")

def test_db():
    print(db.dialect)
    print(db.get_usable_table_names())
    #print(db.get_table_info())
    print(db.run("SELECT * FROM Artist LIMIT 10;"))
    
"""
1. 将SQLite服务转化为工具
"""

from langchain_ollama import ChatOllama

from langchain_community.agent_toolkits import SQLDatabaseToolkit

def create_tools(llm_model_name):
    """创建工具"""

    llm = ChatOllama(model=llm_model_name,temperature=0, verbose=True)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    tools = toolkit.get_tools()
    print(tools)

    return tools

"""
2. 系统提示词
"""

from langchain_core.messages import SystemMessage

system = """您是设计用于与 SQL 数据库交互的代理。用中文回答问题。
给定一个输入问题，创建一个语法正确的 SQLite 查询来运行，然后查看查询结果并返回答案。
除非用户指定他们希望获得的特定数量的示例，否则请始终将查询限制为最多 5 个结果。
您可以按相关列对结果进行排序，以返回数据库中最有趣的示例。
切勿查询特定表中的所有列，仅询问给定问题的相关列。
您可以使用与数据库交互的工具。
仅使用给定的工具。仅使用工具返回的信息来构建最终答案。
在执行查询之前，您必须仔细检查查询。如果在执行查询时出现错误，请重写查询并重试。

请勿对数据库执行任何 DML 语句（INSERT、UPDATE、DELETE、DROP 等）。

您有权访问以下数据库表： {table_names}

如果您需要过滤专有名词，则必须始终先使用“search_proper_nouns”工具查找过滤值！
不要试图猜测专有名词 - 使用此功能查找类似名称。""".format(
    table_names=db.get_usable_table_names()
)

system_message = SystemMessage(content=system)

"""
3. 智能体
"""

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

def ask(llm_model_name,question):
    """询问智能体"""

    tools = create_tools(llm_model_name)
    llm = ChatOllama(model=llm_model_name,temperature=0, verbose=True)
    agent_executor = create_react_agent(llm, tools, state_modifier=system_message)

    for s in agent_executor.stream(
        {"messages": [HumanMessage(content=question)]}
    ):
        print(s)
        print("----")

def test_model(llm_model_name):
    """测试大模型"""

    questions = [
        "有多少名员工？",
        "哪个国家的顾客花费最多？",
        "描述 PlaylistTrack 表"
    ]

    for question in questions:
        ask(llm_model_name,question)

if __name__ == '__main__':

    test_model("qwen2.5")
    test_model("llama3.1")
    test_model("MFDoom/deepseek-r1-tool-calling:7b")