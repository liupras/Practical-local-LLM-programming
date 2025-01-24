#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-24
# @function: 使用大模型实现翻译功能
# @version : V0.5

# 实例化本地大模型
from langchain_ollama.llms import OllamaLLM
model = OllamaLLM(model="llama3.1")

from langchain_core.messages import HumanMessage, SystemMessage

def translate_1(text):
    """将文字翻译为意大利语"""
    messages = [
        SystemMessage("Translate the following from English into Italian"),
        HumanMessage(text),
    ]

    return model.invoke(messages)

def translate_1_stream(text):
    """将文字翻译为意大利语，流式输出"""
    messages = [
        SystemMessage("Translate the following from English into Italian"),
        HumanMessage(text),
    ]
    for token in model.stream(messages):
        yield token

from langchain_core.prompts import ChatPromptTemplate

def translate_2(text,language):
    """用提示词模板构建提示词，翻译文字"""

    # 1. system提示词
    system_template = "Translate the following from English into {language}"

    # 2. 提示词模板
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )

    # 3. 调用invoke构建提示词
    prompt = prompt_template.invoke({"language": language, "text": text})
    print(prompt.to_messages())

    response = model.invoke(prompt)
    return response

if __name__ == '__main__':
    """
    response = translate_1("Hello, how are you?")
    print(response)
    
    for token in translate_1_stream("Hello, how are you?"):
        print(token, end="|")
    """

    response = translate_2("First up, let's learn how to use a language model by itself.","Chinese")
    print(response)
    