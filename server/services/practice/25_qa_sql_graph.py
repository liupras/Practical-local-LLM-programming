#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-20
# @function: sql问答
# @version : V0.5
# @Description ：sql问答。

# https://python.langchain.com/docs/tutorials/sql_qa/


"""
确定文件位置
"""

import os

# 获取当前执行的程序文件的文件夹路径
current_folder = os.path.dirname(os.path.abspath(__file__))

db_file_path = os.path.join(current_folder, 'assert/Chinook.db')

"""
1. 创建SQLite对象
"""

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri(f"sqlite:///{db_file_path}")

def test_db():
    """测试SQLite数据库"""
    print(db.dialect)
    print(db.get_usable_table_names())
    #print(db.get_table_info())
    print(db.run("SELECT * FROM Artist LIMIT 10;"))


"""
2. 状态
"""

from typing_extensions import TypedDict

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1",temperature=0, verbose=True)

def set_llm(llm_model_name):
    """设置大模型，用于测试不同大模型"""
    global llm 
    llm = ChatOllama(model=llm_model_name,temperature=0, verbose=True)

"""
3. 定义langgraph节点
"""

from typing_extensions import Annotated

class QueryOutput(TypedDict):
    """生成的SQL查询语句"""

    query: Annotated[str, ..., "Syntactically valid SQL query."]



# 提示词

system = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

You have access to the following tables: {table_names}
""".format(
    table_names=db.get_usable_table_names(),
    dialect=db.dialect
)

from langchain_core.prompts import ChatPromptTemplate
query_prompt_template = ChatPromptTemplate.from_messages([
    ("system", system),
    ("user", "Question:{input}")
])

def test_prompt():
    """测试提示词"""
    assert len(query_prompt_template.messages) == 1
    query_prompt_template.messages[0].pretty_print()

def write_query(state: State):
    """根据问题生成SQL查询语句"""
    prompt = query_prompt_template.invoke(
        {
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    print(f'Query is:\n{result["query"]}')
    return {"query": result["query"]}


from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

def execute_query(state: State):
    """执行SQL查询"""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    result = execute_query_tool.invoke(state["query"])
    print(f'Result is:\n{result}')
    return {"result": result}


def generate_answer(state: State):
    """使用检索到的信息作为上下文来回答问题。"""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    print(f'answer is:\n{response.content}')
    return {"answer": response.content}

"""
4. langgraph链
"""

from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

def ask(question):
    """问答"""
    for step in graph.stream(
        {"question": question}, stream_mode="updates"
    ):
        print(step)

def test_model(llm_model_name):
    """测试大模型"""

    print(f'============{llm_model_name}==========')

    set_llm(llm_model_name)

    questions = [
        "How many Employees are there?",
        "Which country's customers spent the most?",
        "Describe the PlaylistTrack table",     # 区分大小写，待改进。比如：用 PlaylistTrack 可以工作，但是用 playlisttrack 不准确
    ]

    for question in questions:
        write_query({"question": question})

    for question in questions:
        ask({"question": question})



if __name__ == '__main__':
    
    from utils import show_graph
    show_graph(graph)

    test_db()
    test_prompt()  

    test_model("qwen2.5")
    test_model("llama3.1")
    #test_model("deepseek-r1")
    