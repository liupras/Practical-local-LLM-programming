#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-13
# @function: 给文本打标签/分类
# @version : V0.5
# @Description ：给文本打标签/分类。

# 参考：https://python.langchain.com/docs/tutorials/classification/

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1",temperature=0.2,verbose=True)

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

def simple_control(s):

    tagging_prompt = ChatPromptTemplate.from_template(
        """
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Classification' function.

    Passage:
    {input}
    """
    )

    # 指定​​ Pydantic 模型控制返回内容格式
    class Classification(BaseModel):
        sentiment: str = Field(description="The sentiment of the text")
        aggressiveness: int = Field(
            description="How aggressive the text is on a scale from 1 to 10"
        )
        language: str = Field(description="The language the text is written in")


    llm_structured = llm.with_structured_output(Classification)
    prompt = tagging_prompt.invoke({"input": s})
    response = llm_structured.invoke(prompt)

    return response.model_dump()

def finer_control(s):    
    """
    官网使用OpenAI，我们使用的是本地大模型。
    直接用官网的代码效果不好：sentiment无法按预期标记出happy,neutral,sad，依然只能标记出：positive、negative；aggressiveness的值一直为0。
    """

    # 指定​​ Pydantic 模型控制返回内容格式
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


    llm_structured = llm.with_structured_output(Classification)

    prompt = tagging_prompt.invoke({"input": s})
    response = llm_structured.invoke(prompt)
    return response.model_dump()


if __name__ == '__main__':
    
    s = "I'm incredibly glad I met you! I think we'll be great friends!"
    result = simple_control(s)
    print(f'result:\n{result}')
    
    s = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
    result = simple_control(s)
    print(f'result:\n{result}')
    
    s = "I'm incredibly glad I met you! I think we'll be great friends!"
    result = finer_control(s)
    print(f'finer_control result:\n{result}')
    
    s = "Weather is ok here, I can go outside without much more than a coat"
    result = finer_control(s)
    print(f'finer_control result:\n{result}')

    s="今天的天气糟透了，我什么都不想干！"
    result = finer_control(s)
    print(f'finer_control result:\n{result}')