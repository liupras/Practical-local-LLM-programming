#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-05
# @function: 生成知识库
# @version : V0.5
# @Description ：将csv数据矢量化存储。

import os

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)

persist_directory = os.path.join(current_dir,'db_law')
model = 'llama3.1'

# 安装 langchain_chroma 时需要C++编译器，可以安装Visual Studio
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import CSVLoader
from tqdm import tqdm

def embed_documents_in_batches(documents, batch_size=10):
    """
    按批次嵌入，可以跟踪进度。
    vectordb会自动持久化存储在磁盘。
    """
    vectordb = Chroma(persist_directory=persist_directory,embedding_function=OllamaEmbeddings(model=model))
    for i in tqdm(range(0, len(documents), batch_size), desc="嵌入进度"):
        batch = documents[i:i + batch_size]
        # 从文本块生成嵌入，并将嵌入存储在Chroma向量数据库中，同时设置数据库持久化路径。
        # 耗时较长，需要耐心等候...我执行一次要花费20多分钟。
        vectordb.add_documents(batch)


# 对文本矢量化并存储在本地
def create():

    data_file = os.path.join(current_dir,'assert/law.csv')

    loader = CSVLoader(file_path=data_file,
                       csv_args={"delimiter": "#"},
                       autodetect_encoding=True)
    docs = loader.load()
    print(f'加载文件成功，第一个文件内容：{docs[0]}')

    # 用于将长文本拆分成较小的段，便于嵌入和大模型处理。
    # 每个文本块的最大长度是1000个字符，拆分的文本块之间重叠部分为200。
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)   

    # 耗时较长，需要耐心等候...
    embed_documents_in_batches(texts,batch_size=3)


if __name__ == '__main__':
    create()
 