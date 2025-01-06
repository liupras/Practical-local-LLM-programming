#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-06
# @function: 测试llama3.1 8b模型
# @version : V0.5

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

model_name = "llama3.1"

def ask_me(question):

    template = """Question: {question}

    Answer: Let's think step by step.

    请用简体中文回复。
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model=model_name)
    chain = prompt | model
    result = chain.invoke({"question": question})
    return result

if __name__ == '__main__':
    print(ask_me("你是谁？你有什么本事？"))