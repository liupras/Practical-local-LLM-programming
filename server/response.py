#coding=utf-8

#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2024-12-27
# @Description: 定义API的标准响应
# @version : V0.5

from pydantic import BaseModel, Field

from enum import Enum

# 操作结果枚举
class code_enum(str,Enum):
    OK = 'ok'
    ERR = 'error'

# API返回的消息体
class response_model(BaseModel):
    code: code_enum = Field(description="操作结果" )
    desc: str = Field(description="具体内容" )