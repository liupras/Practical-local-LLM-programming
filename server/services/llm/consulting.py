#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-06
# @function: 基于langchian和实现的对话式RAG(RAG，Retrieval Augmented Generation,即：增强生成)实现知识问答
# @version : V0.5
# @Description ：在问答的过程中，系统自动存储以往的问题和答案，产生“记忆”功能，提升会话体验。

import os

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)

persist_directory = 'db_law'
model = 'llama3.1'

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory


# 返回本地大模型
def get_llm():

    # temperature：用于控制生成语言模型中生成文本的随机性和创造性。
    # 当temperature值较低时，模型倾向于选择概率较高的词，生成的文本更加保守和可预测，但可能缺乏多样性和创造性。
    # 当temperature值较高时，模型选择的词更加多样化，可能会生成更加创新和意想不到的文本，但也可能引入语法错误或不相关的内容。
    # 当需要模型生成明确、唯一的答案时，例如解释某个概念，较低的temperature值更为合适；如果目标是为了产生创意或完成故事，较高的temperature值可能更有助于生成多样化和有趣的文本。
    return ChatOllama(model=model,temperature=0.1,verbose=True)

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
store = {}

from langchain.schema import BaseMessage
MAX_HISTORY_SIZE = 20

# 扩展的聊天历史记录类。可以限制聊天记录的最大长度。
# max_size:设置为偶数。因为User和AI的消息会分别记录为1条，设置为偶数后，User和AI才会成对。
class LimitedChatMessageHistory(ChatMessageHistory):
    _max_size: int
    def __init__(self, max_size: int):        
        super().__init__()       
        self._max_size = max_size 

    def add_message(self, message: BaseMessage):
        super().add_message(message)
        #print(f'记录新消息:{message}')
        # 保持聊天记录在限制范围内
        if len(self.messages) > self._max_size:
            print('消息超限，马上压缩！')
            self.messages = self.messages[-self._max_size:]

# 在会话中记录历史聊天记录
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = LimitedChatMessageHistory(max_size=MAX_HISTORY_SIZE)
    return store[session_id]

def get_retriever():
    
    # 使用本地矢量数据库创建矢量数据库实例
    vectorstore = Chroma(persist_directory=persist_directory, 
                         embedding_function=OllamaEmbeddings(model=model))

    # 处理基于向量数据库的查询回答任务
    return vectorstore.as_retriever()

def get_history_aware_retriever():
    # 构建检索器，将问题放在特定的上下文中进行考虑和回答。
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    llm = get_llm()
    retriever = get_retriever()
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

def get_conversational_rag_chain(): 

    history_aware_retriever = get_history_aware_retriever()

    # 将检索器纳入问答链，回答问题 
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        " If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    llm = get_llm()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain

# 带有历史记录的聊天方法
# 显然，chat_history可以让模型更能“理解”上下文，做出更加妥帖的回答。
def consult(query,session_id):   

    conversational_rag_chain = get_conversational_rag_chain()    

    # 调用链，返回结果
    response = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )
    return response["answer"]

if __name__ == '__main__':

    query1 = "你知道中华人民共和国产品质量法么？"
    query2 = "请用一段文字概括一下它的内容。"
    query3 = "在生产、销售的产品中掺杂、掺假 违反了哪个法律？哪个条款？"
    query4 = "下面的问题与中华人民共和国产品质量法无关。宣扬邪教、迷信 违反了哪个法律？哪个条款？"
    session_id = "liu123"

    # 测试chat方法
    print (consult(query1, session_id))
    print (consult(query2, session_id))
    print (consult(query3, session_id))
    print (consult(query4, session_id))
  
    # 查看聊天历史记录
    print("显示聊天历史记录:")
    for message in store[session_id].messages:
        if isinstance(message, AIMessage):
            prefix = "AI"
        else:
            prefix = "User"

        print(f"{prefix}: {message.content}\n")