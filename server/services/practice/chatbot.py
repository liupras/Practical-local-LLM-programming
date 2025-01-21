#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-06
# @function: 与本地大模型聊天，自动记录聊天历史
# @version : V0.5
# @Description ：在问答的过程中，系统自动存储以往的问题和答案，产生“记忆”功能，提升会话体验。

from langchain_ollama import ChatOllama

# llama3.1  EntropyYue/chatglm3
model_name = "llama3.1"

# 返回本地大模型
def get_llm():

    # temperature：用于控制生成语言模型中生成文本的随机性和创造性。
    # 当temperature值较低时，模型倾向于选择概率较高的词，生成的文本更加保守和可预测，但可能缺乏多样性和创造性。
    # 当temperature值较高时，模型选择的词更加多样化，可能会生成更加创新和意想不到的文本，但也可能引入语法错误或不相关的内容。
    # 当需要模型生成明确、唯一的答案时，例如解释某个概念，较低的temperature值更为合适；如果目标是为了产生创意或完成故事，较高的temperature值可能更有助于生成多样化和有趣的文本。
    return ChatOllama(model=model_name,temperature=0.3,verbose=True)

from common.LimitedChatMessageHistory import SessionHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

session_history = SessionHistory(max_size=60)

# 处理聊天历史
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return session_history.process(session_id)

from langchain_core.messages import HumanMessage

from langchain_core.output_parsers import StrOutputParser
def get_history_chain():
    chain = get_llm() | StrOutputParser()
    return RunnableWithMessageHistory(chain, get_session_history)     

with_message_history = get_history_chain()

def chat(human_message,session_id):
    """
    聊天
    """
   
    response = with_message_history.invoke(
        [HumanMessage(content=human_message)],
        config={"configurable": {"session_id": session_id}},
    )

    return response


if __name__ == '__main__':

    session_id = "liu123"

    print (chat("计算 5 加 3 乘以 2 的结果是多少？", session_id))
    '''
    print (chat("你知道x-space的老板马斯克么？", session_id))
    print (chat("他出生在哪个国家？", session_id))
    print (chat("他和特朗普是什么关系？", session_id))
    '''
    #session_history.print_history(session_id)
