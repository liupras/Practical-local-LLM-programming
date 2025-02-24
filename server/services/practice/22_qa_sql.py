#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-19
# @function: sql问答
# @version : V0.5
# @Description ：sql问答。

# https://python.langchain.com/v0.2/docs/tutorials/sql_qa/
# https://sqlitestudio.pl/

"""
确定文件位置
"""
import os

# 获取当前执行的程序文件的文件夹路径
current_folder = os.path.dirname(os.path.abspath(__file__))

db_file_path = os.path.join(current_folder, 'assert/Chinook.db')

"""
1. 测试数据库
"""

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri(f"sqlite:///{db_file_path}")

def test_db():
    """测试数据库"""

    print(db.dialect)
    print(db.get_usable_table_names())
    #print(db.get_table_info())
    print(db.run("SELECT * FROM Artist LIMIT 1;"))

"""
2. 执行SQL
"""
from langchain_ollama import ChatOllama
from langchain.chains import create_sql_query_chain

def execute_query(llm_model_name,question: str):
    """把问题转换为SQL语句并执行"""
    
    llm = ChatOllama(model=llm_model_name,temperature=0, verbose=True)
    chain = create_sql_query_chain(llm, db)
    print(chain.get_prompts()[0].pretty_print())

    # 转化问题为SQL
    response = chain.invoke({"question": question})
    print(f'response SQL is:\n{response}')

    # 执行SQL
    result = db.run(response)
    print(f'result is:\n{result}')

from langchain_community.tools import QuerySQLDataBaseTool

def execute_query_2(llm_model_name,question: str):
    """把问题转换为SQL语句并执行"""

    llm = ChatOllama(model=llm_model_name,temperature=0, verbose=True)
    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm, db)
    chain = write_query | execute_query
    response = chain.invoke({"question": question})
    print(f'response SQL is:\n{response}')

"""
3. 回答问题
"""

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

def ask(llm_model_name,question: str):
    answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
    )

    llm = ChatOllama(model=llm_model_name,temperature=0, verbose=True)
    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm, db)
    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({"question": question})
    print(f'Answer is:\n{response}')

def test_model(model_name):

    qs = [
        "How many Employees are there?",
        "Which country's customers spent the most?",
        "Describe the PlaylistTrack table."  #区分大小写，待改进。比如：用 PlaylistTrack 可以工作，但是用 playlisttrack 不准确
    ]  
    for q in qs:
        execute_query(model_name,q)
        execute_query_2(model_name,q)
        ask(model_name,q)


if __name__ == '__main__':
    test_db()
    test_model("llama3.1")
    test_model("qwen2.5")    
    test_model("deepseek-r1")
    