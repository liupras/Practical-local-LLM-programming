#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-08
# @function: 多模态识别图片
# @version : V0.5
# @Description ：llama3.1不行。

# https://python.langchain.com/docs/how_to/multimodal_inputs/


import os

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)

image_file_path = os.path.join(current_dir,'assert/nature-boardwalk.jpg')

import base64

def open_img(image_file_path):
    with open(image_file_path, "rb") as image_file:
        # 读取图片的二进制数据
        image_data = image_file.read()
        # 编码为 Base64
        base64_encoded = base64.b64encode(image_data).decode('utf-8')
        return base64_encoded

image_data = open_img(image_file_path)

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1",temperature=0.1,verbose=True)
"""
temperature：用于控制生成语言模型中生成文本的随机性和创造性。
当temperature值较低时，模型倾向于选择概率较高的词，生成的文本更加保守和可预测，但可能缺乏多样性和创造性;
当temperature值较高时，模型选择的词更加多样化，可能会生成更加创新和意想不到的文本，但也可能引入语法错误或不相关的内容。
当需要模型生成明确、唯一的答案时，例如解释某个概念，较低的temperature值更为合适；如果目标是为了产生创意或完成故事，较高的temperature值可能更有助于生成多样化和有趣的文本。
"""

from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "describe the weather in this image"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
    ],
)
response = llm.invoke([message])
print(response.content)
