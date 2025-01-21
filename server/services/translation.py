#coding=utf-8

#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-06
# @Description: 使用FastAPI做langchain的API
# @version : V0.5

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM

def translate(language,text):
    # 1. 创建提示词模板
    system_template ='''You are an experienced translator,please use concise words to complete the translation.
    Please only give the translation results.
    ###
    Translate the following into {language}:'''
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', system_template),
        ('user', '{text}')
    ])

    # 2. 创建本地大模型
    model = OllamaLLM(model="llama3.1")

    # 3. 创建解析器
    parser = StrOutputParser()

    # 4. 创建链
    chain = prompt_template | model | parser

    #5. 调用链
    result = chain.invoke({"language": language,"text":text})

    return result
   
if __name__ == '__main__':
    print(translate("简体中文","You are an experienced translator, please use concise words to complete the translation."))