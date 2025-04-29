#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-20
# @function: 总结文本
# @version : V0.5
# @Description ：总结文本。

# https://python.langchain.com/docs/tutorials/summarization/

import os 
os.environ['USER_AGENT'] = 'summarize'

from langchain_community.document_loaders import WebBaseLoader

"""
1. 加载文档
"""

loader = WebBaseLoader("http://wfcoding.com/articles/practice/03%E6%9C%AC%E5%9C%B0%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%BC%96%E7%A8%8B%E5%AE%9E%E6%88%98/",encoding='utf-8')
docs = loader.load()

from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5",temperature=0.3, verbose=True)
# llama3.1 不能执行此任务
#llm = ChatOllama(model="llama3.1",temperature=0.3, verbose=True)

"""
2. 用一次调用提取摘要
"""

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 定义提示词
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "请简明扼要地概括以下内容:\\n\\n{context}")
    ]
)

# 初始化 chain
chain = create_stuff_documents_chain(llm, prompt)

def sum_single_llm_call() :    

    # 调用 chain
    result = chain.invoke({"context": docs})
    print(result)

"""
3. 流式输出
"""

def sum_single_llm_call_stream() :    

    for token in chain.stream({"context": docs}):
        print(token, end="|")

if __name__ == '__main__':
    #sum_single_llm_call()
    sum_single_llm_call_stream()
