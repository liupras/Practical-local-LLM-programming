#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-19
# @function: sql问答
# @version : V0.5
# @Description ：sql问答。

# https://python.langchain.com/v0.2/docs/tutorials/sql_qa/
# https://sqlitestudio.pl/

import os

# 获取当前文件的路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件所在文件夹的上一级文件夹路径
parent_folder = os.path.dirname(os.path.dirname(current_file_path))

db_file_path = os.path.join(parent_folder, 'assert/Chinook.db')

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri(f"sqlite:///{db_file_path}")

def test_db():
    print(db.dialect)
    print(db.get_usable_table_names())
    #print(db.get_table_info())
    print(db.run("SELECT * FROM Artist LIMIT 10;"))

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1",temperature=0, verbose=True)

# Convert question to SQL query
from langchain.chains import create_sql_query_chain

def test_query(question: str):
    """
    使用llama3.1和codellama都能工作
    """
    
    chain = create_sql_query_chain(llm, db)
    response = chain.invoke({"question": question})
    print(f'response SQL is:\n{response}')
    result = db.run(response)
    print(f'result is:\n{result}')

# Execute SQL query
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
chain = write_query | execute_query

def test_query_2(question: str):
    response = chain.invoke({"question": question})
    print(f'response SQL is:\n{response}')

# Answer the question

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

def ask(question: str):
    answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
    )

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

if __name__ == '__main__':
    #test_db()
    #test_prompt()

    question = "How many Employees are there?"
    #test_query_2(question)
    #ask(question)

    question = "Which country's customers spent the most?"
    #test_query(question)
    #test_query_2(question)
    #ask(question)

    question = "Describe the PlaylistTrack table"       #区分大小写，待改进。比如：用 PlaylistTrack 可以工作，但是用 playlisttrack 不准确
    ask(question)
    