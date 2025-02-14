#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-02-11
# @function: 在本地处理矢量数据库。
# @version : V0.5
# @Description ：使用Chroma，处理本地适量数据库。

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter

class LocalVectorDBChroma:
    """使用Chroma在本地处理适量数据库"""

    def __init__(self,model_name,persist_directory,delimiter = ","):
        self._embedding = OllamaEmbeddings(model=model_name)
        self._persist_directory = persist_directory
        self._delimiter = delimiter

    def get_vector_store(self):
        return Chroma(persist_directory=self._persist_directory,embedding_function=self._embedding)

    def embed_documents_in_batches(self,documents,batch_size=3):
        """
        按批次嵌入，可以显示进度。
        vectordb会自动持久化存储在磁盘。
        """
        
        vectordb = Chroma(persist_directory=self._persist_directory,embedding_function=self._embedding)
        for i in tqdm(range(0, len(documents), batch_size), desc="嵌入进度"):
            batch = documents[i:i + batch_size]

            # 从文本块生成嵌入，并将嵌入存储在本地磁盘。
            vectordb.add_documents(batch)

    def embed_csv(self,src_file_path):
        """嵌入csv"""

        from langchain_community.document_loaders import CSVLoader
        
        loader = CSVLoader(file_path=src_file_path,
                       csv_args={"delimiter": self._delimiter},
                       autodetect_encoding=True)
        docs = loader.load()

        # 用于将长文本拆分成较小的段，便于嵌入和大模型处理。     
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        """
        chunk_size: 每个文本块的最大长度/字符数
        chunk_overlap: 拆分的文本块之间重叠字符数
        """
        documents = text_splitter.split_documents(docs) 

        # 耗时较长，需要耐心等候...
        self.embed_documents_in_batches(documents)

    def embed_webpage(self,url):
        """嵌入网页"""

        from langchain_community.document_loaders import WebBaseLoader

        loader = WebBaseLoader(url,encoding="utf-8")    # 增加encoding参数防止中文乱码
        docs = loader.load()
        documents = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(docs)

        self.embed_documents_in_batches(documents)
