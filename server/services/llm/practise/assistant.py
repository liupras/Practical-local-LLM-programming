#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-07
# @function: AI助理
# @version : V0.5
# @Description ：修改提示词，大模型秒变助理。

from langchain_ollama import ChatOllama

# 返回本地大模型
def get_llm():

    # temperature：用于控制生成语言模型中生成文本的随机性和创造性。
    # 当temperature值较低时，模型倾向于选择概率较高的词，生成的文本更加保守和可预测，但可能缺乏多样性和创造性。
    # 当temperature值较高时，模型选择的词更加多样化，可能会生成更加创新和意想不到的文本，但也可能引入语法错误或不相关的内容。
    # 当需要模型生成明确、唯一的答案时，例如解释某个概念，较低的temperature值更为合适；如果目标是为了产生创意或完成故事，较高的temperature值可能更有助于生成多样化和有趣的文本。
    return ChatOllama(model="llama3.1",temperature=0.3,verbose=True)


from common.LimitedChatMessageHistory import SessionHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

session_history = SessionHistory(max_size=20)

# 处理聊天历史
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return session_history.process(session_id)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_history_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt |  get_llm()
    return RunnableWithMessageHistory(chain, get_session_history)

with_message_history = get_history_chain()

from langchain_core.messages import HumanMessage

def chat(human_message,session_id):
    """
    助理
    """

    response = with_message_history.invoke(
        [HumanMessage(content=human_message)],
        config={"configurable": {"session_id": session_id}},
    )

    return response.content

if __name__ == '__main__':

    session_id = "liu123"

    # 测试chat方法
    print (chat("你知道x-space的老板马斯克么？", session_id))
    print (chat("他出生在哪个国家？", session_id))
    print (chat("他和特朗普是什么关系？", session_id))
  
    session_history.print_history(session_id)