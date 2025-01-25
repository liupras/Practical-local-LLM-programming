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

def get_file_path():
    """获取文件路径。
    用这种方式获取资源文件路径比较安全。
    """

    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)

    file_path = os.path.join(current_dir,'assert/nke-10k-2023.pdf')

    return file_path

def load_file(file_path):
    """加载pdf文件"""    

    # Loading documents
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(file_path)

    docs = loader.load()

    print(f'加载文件成功，总文本数:{len(docs)}')

    # PyPDFLoader loads one Document object per PDF page. The first page is at index 0.
    print(f"page one:\n{docs[0].page_content[:200]}\n")
    print(f'page one metadata:\n{docs[0].metadata}')

    return docs


def split_text(docs):
    """分割文档"""

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    print(f"Number of splits: {len(all_splits)}")  

    return all_splits

# Embeddings

from langchain_ollama.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
"""
nomic-embed-text: 一个高性能开放嵌入模型，只有27M，具有较大的标记上下文窗口。
在做英文的嵌入和检索时，明显比llama3.1要好，可惜做中文不行。
"""    

def get_vector_store():
    """获取内存矢量数据库"""

    from langchain_core.vectorstores import InMemoryVectorStore

    vector_store = InMemoryVectorStore(embeddings)

    file_path = get_file_path()
    docs = load_file(file_path)
    all_splits = split_text(docs)
    _ = vector_store.add_documents(documents=all_splits)

    return vector_store

def similarity_search(query):
    """内存矢量数据库检索测试"""

    vector_store = get_vector_store()
    results = vector_store.similarity_search(query)
    return results

def similarity_search_with_score(query):
    """内存矢量数据库检索测试
    返回文档评分，分数越高，文档越相似。
    """
    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(query)
    return results

def embed_query(query):
    """嵌入查询测试"""

    embedding = embeddings.embed_query(query)

    vector_store = get_vector_store()
    results = vector_store.similarity_search_by_vector(embedding)
    return results


# Retrievers
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain


@chain
def retriever(query: str) -> List[Document]:
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=1)


def retriever_batch_1(query:List[str]):
    r = retriever.batch(query)
    return r


def retriever_batch_2(query:List[str]):
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    r = retriever.batch(query)
    return r

if __name__ == '__main__':
    '''
    load_file(get_file_path())
    
    results = similarity_search("How many distribution centers does Nike have in the US?")
    print(f'similarity_search results[0]:\n{results[0]}')
    
    results = similarity_search_with_score("What was Nike's revenue in 2023?")
    doc, score = results[0]
    print(f"Score and doc: {score}\n{doc}")
   
    results = embed_query("How were Nike's margins impacted in 2023?")
    print(f'embed_query results[0]:\n{results[0]}') 
    '''
    query = [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ]

    results = retriever_batch_1(query)
    print(f'retriever.batch 1:\n{results}')
    results = retriever_batch_2(query)
    print(f'retriever.batch 2:\n{results}')



