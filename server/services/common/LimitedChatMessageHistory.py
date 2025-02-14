#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-06
# @function: 扩展的聊天历史记录类。
# @version : V0.5
# @Description ：可以限制聊天记录的最大长度。max_size:设置为偶数。因为User和AI的消息会分别记录为1条，设置为偶数后，User和AI才会成对。

'''
在 https://python.langchain.com/v0.2/docs/tutorials/chatbot/ 中有使用trim_messages对消息历史进行裁剪的例子
但是这里依然需要用大模型来计算token，通过计算结果进行裁剪，比较耗费资源。

本例的方法没那么智能，也可能有时候会突破token大小限制出错，但是我想已经能解决绝大部分问题了。
'''

from langchain.schema import BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory

class MessageHistory(ChatMessageHistory):
    """
    扩展的聊天历史记录类。可以限制聊天记录的最大长度。

    Args:
        max_size: 设置为偶数。因为User和AI的消息会分别记录为1条，设置为偶数后，User和AI才会成对。
    """

    def __init__(self, max_size: int):        
        super().__init__()       
        self._max_size = max_size 

    def add_message(self, message: BaseMessage):
        super().add_message(message)  

        # 保持聊天记录在限制范围内
        if len(self.messages) > self._max_size:
            print('消息超限，马上压缩！')
            self.messages = self.messages[-self._max_size:]

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage

class SessionHistory(object):
    """
    处理消息历史
    """
    def __init__(self,max_size: int):
        super().__init__()
        self._max_size = max_size
        self._store = {}

    def process(self,session_id: str) -> BaseChatMessageHistory:
        """
        处理聊天历史
        """
        if session_id not in self._store:
            self._store[session_id] = MessageHistory(max_size=self._max_size)
        return self._store[session_id]

    def print_history(self,session_id):
        """
        查看聊天历史记录
        """
        print("显示聊天历史记录...")
        for message in self._store[session_id].messages:
            if isinstance(message, AIMessage):
                prefix = "AI"
            else:
                prefix = "User"

            print(f"{prefix}: {message.content}\n")
