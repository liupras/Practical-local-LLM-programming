#coding=utf-8

#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-06
# @Description: 使用FastAPI和langchain翻译服务API
# @version : V0.5


import sys
import os

# 将上级目录加入path，这样就可以直接引用response等上级目录的模块
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# 导入FastAPI和Pydantic
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from response import response_model,code_enum

# 创建一个FastAPI实例
app = FastAPI()

from translation import translate

class query_model_translation(BaseModel):
    lang: str = Field(min_length=2, max_length=20, description="语言名称" )
    text: str = Field(min_length=2, max_length=500, description="待翻译的文本" )


"""
!注意：设置端点时，建议养成都不加 / 的风格。
在使用API网关时，如果从API网关传过来的路径是以 / 结尾的话，因为和此端点路径不一致，此端点会自动返回301重定向，导致客户端发生400错误。
"""

@app.post("/trans/v1", response_model=response_model)
async def translate(query: query_model_translation,request: Request):
    """
    翻译文本。

    参数:
    - query_model_translation: 翻译请求内容。

    返回:
    - response: 对象
    """
    userid = request.headers.get("userid")
    print(f"userid: {userid}")
    
    try:
        r = translate(query.lang.strip(),query.text.strip())
        return response_model(code=code_enum.OK,desc=r)
    except Exception as e:
        return response_model(code=code_enum.ERR,desc=str(e))

from consulting import consult

class query_model_consult(BaseModel):
    question: str = Field(min_length=2, max_length=1000, description="咨询的问题内容" )

@app.post("/consult/v1", response_model=response_model)
async def consult(query: query_model_consult,request: Request):
    """
    咨询专家

    参数:
    - query_model_consult: 翻译请求内容。

    返回:
    - response: 对象
    """
    userid = request.headers.get("userid")
    if userid is None:
        userid = "test"
    print(f"userid: {userid}")
    try:
        r = consult(query.question.strip(),userid)
        return response_model(code=code_enum.OK,desc=r)
    except Exception as e:
        return response_model(code=code_enum.ERR,desc=str(e))

import uvicorn

if __name__ == '__main__':
    """
    交互式API文档地址：
    http://127.0.0.1:5001/docs/ 
    http://127.0.0.1:5001/redoc/
    """
    
    uvicorn.run(app, host="0.0.0.0", port=5001)
   