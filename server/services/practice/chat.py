#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-06
# @function: 测试llama3.1 8b模型
# @version : V0.5

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# llama3.1  EntropyYue/chatglm3
model_name = "EntropyYue/chatglm3"

def ask(question):

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
    print(ask("你是谁？你有什么本事？"))