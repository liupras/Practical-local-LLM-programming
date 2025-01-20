#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-19
# @function: sql问答agent
# @version : V0.5
# @Description ：sql问答agent。

# https://python.langchain.com/v0.2/docs/tutorials/sql_qa/
# https://sqlitestudio.pl/

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

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1",temperature=0, verbose=True)

from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

print(tools)

# System Prompt
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

If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool!
Do not try to guess at the proper name - use this function to find similar ones.""".format(
    table_names=db.get_usable_table_names()
)

system_message = SystemMessage(content=system)

# Initializing agent

from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(llm, tools, state_modifier=system_message)

from langchain_core.messages import HumanMessage

def ask(question):
    for s in agent_executor.stream(
        {"messages": [HumanMessage(content=question)]}
    ):
        print(s)
        print("----")

from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

from langchain_chroma import Chroma

persist_directory = os.path.join(current_folder,'assert/db_artists_albums')
vectordb = Chroma(persist_directory=persist_directory,embedding_function=embeddings)

from tqdm import tqdm

def embed_texts_in_batches(documents, batch_size=10):
    """
    按批次嵌入，可以跟踪进度。
    vectordb会自动持久化存储在磁盘。
    """
    
    for i in tqdm(range(0, len(documents), batch_size), desc="嵌入进度"):
        batch = documents[i:i + batch_size]
        # 从文本块生成嵌入，并将嵌入存储在Chroma向量数据库中，同时设置数据库持久化路径。
        # 耗时较长，需要耐心等候...
        vectordb.add_texts(batch)


def create_db():
    import ast
    import re

    def query_as_list(db, query):
        res = db.run(query)
        res = [el for sub in ast.literal_eval(res) for el in sub if el]
        res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
        return list(set(res))

    artists = query_as_list(db, "SELECT Name FROM Artist")
    print(f'artists:\n{artists[:5]}\n') 
    albums = query_as_list(db, "SELECT Title FROM Album")
    print(f'albums:\n{albums[:5]}\n')

    documents = artists + albums
    embed_texts_in_batches(documents)
    print('db_artists_albums persisted.')

def create_retriever_tool():

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    from langchain.agents.agent_toolkits import create_retriever_tool
    description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
    valid proper nouns. Use the noun most similar to the search."""
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_proper_nouns",
        description=description,
    )

    return retriever_tool

retriever_tool = create_retriever_tool()

tools.append(retriever_tool)

agent_executor_2 = create_react_agent(llm, tools, messages_modifier=system_message)

def ask_2(question):
    for s in agent_executor_2.stream(
        {"messages": [HumanMessage(content=question)]}
    ):
        print(s)
        print("----")

if __name__ == '__main__':
    test_db()

    question = "How many Employees are there?"
    #ask(question) 

    question = "Which country's customers spent the most?"
    #ask(question) 
    
    question = "Describe the PlaylistTrack table"   #区分大小写，待改进。比如：用 PlaylistTrack 可以工作，但是用 playlisttrack 不行
    #ask(question)

    #create_db()
    #print(retriever_tool.invoke("Alice Chains"))

    question = "How many albums does alis in chain have?"       #llama3.1搞不定
    ask(question)


    