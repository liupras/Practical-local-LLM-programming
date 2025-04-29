#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-04-28
# @function: 使用Map-Reduce对大量文本提取摘要
# @version : V0.5

# https://python.langchain.com/docs/tutorials/summarization/

import os
os.environ['USER_AGENT'] = 'summarize'

llm_model_name = "qwen2.5"

from langchain_ollama import ChatOllama
llm = ChatOllama(model=llm_model_name,temperature=0.3, verbose=True)

"""
1. 准备文档
"""

def load_document(url):
    """加载文档"""

    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader(url,encoding='utf-8')
    doc = loader.load()

    return doc

def split_document(url):
    """分割文档，为Map做准备"""

    doc = load_document(url)
    
    from langchain_text_splitters import CharacterTextSplitter
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(doc)
    print(f"Generated {len(split_docs)} documents.")

    return split_docs

split_docs = split_document("http://www.wfcoding.com/articles/practice/0325/")

"""
2. Map-Reduce提示词
"""
# Map时使用
from langchain_core.prompts import ChatPromptTemplate
map_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "请简明扼要地概括以下内容:\\n\\n{context}")
    ]
)

# Reduce时使用
reduce_template = """
以下是一组摘要：
{docs}
请将这些内容提炼成最终的、综合的主题摘要。
"""
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

"""
3. 确定token数量
"""

from typing import List
from langchain_core.documents import Document
import jieba
def count_tokens(text):
    """返回token数量"""

    tokens = jieba.lcut(text)
    return len(tokens)

def length_function(documents: List[Document]) -> int:
    """获取输入内容的token数量。"""

    return sum(count_tokens(doc.page_content) for doc in documents)

"""
4. 定义状态
"""
import operator
from typing import Annotated, Literal, TypedDict

class OverallState(TypedDict):
    """主体状态
    
    这里我们使用了operator.add，这是因为我们想将所有从各个节点生成的摘要合并到一个列表中。
    """

    contents: List[str]     # 分割后的原始文档列表
    summaries: Annotated[list, operator.add]   # 由原始文档列表生成的摘要列表 
    collapsed_summaries: List[Document]     # 折叠/压缩的文档列表
    final_summary: str      # 最终提取的摘要


class SummaryState(TypedDict):
    """将所有文档“映射”到的节点的状态，以便生成摘要"""

    content: str

"""
5. 定义节点/步骤
"""

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langgraph.graph import END, START, StateGraph
from langgraph.constants import Send

token_max = 1000

async def generate_summary(state: SummaryState):
    """提取一个文档的摘要"""

    prompt = map_prompt.invoke(state["content"])
    response = await llm.ainvoke(prompt)
    return {"summaries": [response.content]}

def map_summaries(state: OverallState):
    """
    【边】把文档列表中的每一个文档map出去

    返回一个 `Send` 对象列表。每个 `Send` 对象包含图中一个节点的名称以及要发送到该节点的状态。
    """

    return [
        Send("generate_summary", {"content": content}) for content in state["contents"]
    ]


def collect_summaries(state: OverallState):
    """收集从map出去的文档提取的摘要
    
    所有摘要放在 collapsed_summaries 中，后面可以对它折叠/压缩，直到摘要小于token_max。
    """

    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }


async def _reduce(input: dict) -> str:
    prompt = reduce_prompt.invoke(input)
    response = await llm.ainvoke(prompt)
    return response.content


async def collapse_summaries(state: OverallState):
    """折叠/压缩摘要"""

    doc_lists = split_list_of_docs(
        state["collapsed_summaries"], length_function, token_max
    )

    # 使用reduce提示词折叠/压缩摘要列表
    results = []
    for doc_list in doc_lists:
        results.append(await acollapse_docs(doc_list, _reduce))

    return {"collapsed_summaries": results}


def should_collapse(
    state: OverallState,
) -> Literal["collapse_summaries", "generate_final_summary"]:
    """【边】确定我们是否应该折叠/压缩摘要"""

    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"


async def generate_final_summary(state: OverallState):
    """生成最终摘要"""

    response = await _reduce(state["collapsed_summaries"])
    return {"final_summary": response}


def create_graph():
    """构建langgraph图"""

    graph = StateGraph(OverallState)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("collapse_summaries", collapse_summaries)
    graph.add_node("generate_final_summary", generate_final_summary)

    # Edges:
    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", END)

    app = graph.compile()
    return app

async def summarize():
    """提取摘要"""

    app = create_graph()

    async for step in app.astream(
        {"contents": [doc.page_content for doc in split_docs]},
        {"recursion_limit": 10},
    ):
        step_keys = list(step.keys())
        print(step_keys)
        if 'generate_final_summary' in step_keys:
            print(step['generate_final_summary'])


if __name__ == '__main__':

    """
    from utils import show_graph
    app = create_graph()
    show_graph(app)
    """

    import asyncio
    asyncio.run(summarize())

    
