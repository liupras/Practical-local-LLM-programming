#coding=utf-8

#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-14
# @function: 利用本地大模型测试矢量数据库
# @Description: 使用 nomic-embed-text 做英文嵌入检索很好，使用 llama3.1 效果一般
# @version : V0.5

# https://python.langchain.com/docs/tutorials/retrievers/

import os
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)

file_path = os.path.join(current_dir,'assert/nke-10k-2023.pdf')

# Loading documents
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path)

docs = loader.load()

print(f'加载文件成功，总文本数:{len(docs)}')

# PyPDFLoader loads one Document object per PDF page. For each, we can easily access
print(f"page one:\n{docs[0].page_content[:200]}\n")
print(f'page one metadata:\n{docs[0].metadata}')

# Splitting

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(f"Number of splits: {len(all_splits)}")  

# Embeddings

from langchain_community.embeddings import OllamaEmbeddings

# nomic-embed-text  llama3.1    EntropyYue/chatglm3
embeddings = OllamaEmbeddings(model="nomic-embed-text")
"""
nomic-embed-text: 一个高性能开放嵌入模型，只有27M，具有较大的标记上下文窗口。
在做英文的嵌入和检索时，明显比llama3.1要好，可惜做中文不行。
"""    

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1",temperature=0.3,verbose=True)
"""
temperature：用于控制生成语言模型中生成文本的随机性和创造性。
当temperature值较低时，模型倾向于选择概率较高的词，生成的文本更加保守和可预测，但可能缺乏多样性和创造性。
当temperature值较高时，模型选择的词更加多样化，可能会生成更加创新和意想不到的文本，但也可能引入语法错误或不相关的内容。
当需要模型生成明确、唯一的答案时，例如解释某个概念，较低的temperature值更为合适；如果目标是为了产生创意或完成故事，较高的temperature值可能更有助于生成多样化和有趣的文本。
"""

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

print(f'results[0]:\n{results[0]}')  # print(results[0])

# Return scores

# Note that providers implement different scores; the score here
# is a distance metric that varies inversely with similarity.

results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(f"Score and doc: {score}\n{doc}")

# Return documents based on similarity to an embedded query
embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")

results = vector_store.similarity_search_by_vector(embedding)
print(f'results[0]:\n{results[0]}') 

# Retrievers
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain


@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


r = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)

print(f'retriever.batch:\n{r}')

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

r = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)

print(f'retriever.batch:\n{r}')

