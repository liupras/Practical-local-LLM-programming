#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-13
# @function: 给文本打标签/分类
# @version : V0.5
# @Description ：给文本打标签/分类。

# https://python.langchain.com/docs/tutorials/classification/

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

def simple_control():

    tagging_prompt = ChatPromptTemplate.from_template(
        """
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Classification' function.

    Passage:
    {input}
    """
    )

    class Classification(BaseModel):
        sentiment: str = Field(description="The sentiment of the text")
        aggressiveness: int = Field(
            description="How aggressive the text is on a scale from 1 to 10"
        )
        language: str = Field(description="The language the text is written in")



    llm = ChatOllama(model="llama3.1",temperature=0,verbose=True).with_structured_output(
        Classification
    )

    inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
    prompt = tagging_prompt.invoke({"input": inp})
    response = llm.invoke(prompt)

    print(f'response:\n{response}') 

    # dictionary output
    inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
    prompt = tagging_prompt.invoke({"input": inp})
    response = llm.invoke(prompt)

    print(f'response:\n{response.model_dump()}') 


def finer_control():    
    """
    官网使用OpenAI，我们使用的是本地大模型。
    直接用官网的代码效果不好：sentiment无法按预期标记出happy,neutral,sad，依然只能标记出：positive、negative；aggressiveness的值一直为0。
    mistral貌似比llama3.1更好一点，能更准确的进行标记。
    """

    class Classification(BaseModel):
        sentiment: str = Field(description="The sentiment of the text,it must be one of happy,neutral,sad")
        aggressiveness: int = Field(description="The aggressive of the text,it must be one of 1,2,3,4,5,6,7,8,9,10,the higher the number the more aggressive")
        language: str = Field(description="The language the text is written in,it must be one of English,Spanish,Chinese")


    tagging_prompt = ChatPromptTemplate.from_template(
        """
        Extract the desired information from the following passage.

        Only extract the properties mentioned in the 'Classification' function.

        Passage:
        {input}
        """
    )


    llm = ChatOllama(model="llama3.1",temperature=0.5,verbose=True).with_structured_output(
        Classification
    )

    inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
    prompt = tagging_prompt.invoke({"input": inp})
    response = llm.invoke(prompt)
    print(f'response:\n{response.model_dump()}') 

    inp = "Weather is ok here, I can go outside without much more than a coat"
    prompt = tagging_prompt.invoke({"input": inp})
    response =llm.invoke(prompt)
    print(f'response:\n{response}') 

    inp = "今天的天气糟透了，我什么都不想干！"
    prompt = tagging_prompt.invoke({"input": inp})
    response =llm.invoke(prompt)
    print(f'response:\n{response}') 

if __name__ == '__main__':
    finer_control()