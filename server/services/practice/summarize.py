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

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

from langchain_ollama import ChatOllama
llm = ChatOllama(model="EntropyYue/chatglm3",temperature=0.3, verbose=True)
# llama3.1 不能执行此任务
#llm = ChatOllama(model="llama3.1",temperature=0.3, verbose=True)

# Stuff: summarize in a single LLM call

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Define prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Write a concise summary of the following:\\n\\n{context}")
    ]
)


# Instantiate chain
chain = create_stuff_documents_chain(llm, prompt)

def sum_single_llm_call() :    

    # Invoke chain
    result = chain.invoke({"context": docs})
    print(result)

def sum_single_llm_call_stream() :    

    for token in chain.stream({"context": docs}):
        print(token, end="|")

# Map
map_prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following:\\n\\n{context}")]
)

# Reduce
# Also available via the hub: `hub.pull("rlm/reduce-prompt")`
reduce_template = """
The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated summary
of the main themes.
"""

reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

# Orchestration via LangGraph

from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)
print(f"Generated {len(split_docs)} documents.")



if __name__ == '__main__':
    #sum_single_llm_call()
    sum_single_llm_call_stream()
