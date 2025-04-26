#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-21
# @function: 使用图形数据库进行问答
# @version : V0.5
# @Description ：使用图形数据库进行问答。

# 安装neo4j
# 下载地址：https://neo4j.com/deployment-center/
# 安装：https://neo4j.com/docs/operations-manual/current/installation/windows/
# 安装APOC插件：https://github.com/neo4j/apoc/releases/tag/5.26.1

import os
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "neo4j"

from langchain_neo4j import Neo4jGraph

graph = Neo4jGraph()

def create_graph():
    """导入数据，创建图形数据库"""

    # 把movies_small.csv拷贝到neo4j的import文件夹内
    db_file_path = 'file:///movies_small.csv'

    movies_query = """
    LOAD CSV WITH HEADERS FROM 
    '%s'
    AS row
    MERGE (m:Movie {id:row.movieId})
    SET m.released = date(row.released),
        m.title = row.title,
        m.imdbRating = toFloat(row.imdbRating)
    FOREACH (director in split(row.director, '|') | 
        MERGE (p:Person {name:trim(director)})
        MERGE (p)-[:DIRECTED]->(m))
    FOREACH (actor in split(row.actors, '|') | 
        MERGE (p:Person {name:trim(actor)})
        MERGE (p)-[:ACTED_IN]->(m))
    FOREACH (genre in split(row.genres, '|') | 
        MERGE (g:Genre {name:trim(genre)})
        MERGE (m)-[:IN_GENRE]->(g))
    """ % (db_file_path)

    graph.query(movies_query)

    graph.refresh_schema()
    print(graph.schema)    

enhanced_graph = Neo4jGraph(enhanced_schema=True)
print(enhanced_graph.schema)

from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5",temperature=0, verbose=True)   #llama3.1查不出内容；EntropyYue/chatglm3生成的查询有问题报错

# GraphQACypherChain
from langchain_neo4j import GraphCypherQAChain

chain = GraphCypherQAChain.from_llm(
    graph=enhanced_graph, llm=llm, verbose=True, allow_dangerous_requests=True
)

def ask(question:str):
    """询问图数据库内容"""

    response = chain.invoke({"query": question})
    print(f'response:\n{response}')

if __name__ == '__main__':
    
    #create_graph()
    ask("What was the cast of the Casino?")


