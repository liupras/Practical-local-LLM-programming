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

system = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

You have access to the following tables: {table_names}
""".format(
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

    print(f'=========={llm_model_name}==========\n')

    questions = [
        "How many Employees are there?",
        "Which country's customers spent the most?",
        "Describe the PlaylistTrack table"   #区分大小写，待改进。比如：用 PlaylistTrack 可以工作，但是用 playlisttrack 不行
    ]

    for question in questions:
        ask(llm_model_name,question)

if __name__ == '__main__':

    test_model("qwen2.5")

    test_model("llama3.1")    

    test_model("MFDoom/deepseek-r1-tool-calling:7b")