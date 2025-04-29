#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-04-28
# @function: 用websocket显示LLM的输出流
# @version : V0.5

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,AIMessage

model_name = "qwen3"

llm = ChatOllama(model=model_name,temperature=0.3,verbose=True)

def ask(question):  

    result = llm.invoke([HumanMessage(content=question)])
    return result.content

import asyncio
async def ask_stream(question,websocket=None):
    """与大模型聊天，流式输出"""

    for chunk in llm.stream([HumanMessage(content=question)]):
        if isinstance(chunk, AIMessage) and chunk.content !='':
            print(chunk.content,end="^")
            if websocket is not None:
                await websocket.send_json({"reply": chunk.content})
                await asyncio.sleep(0.1)    # sleep一下后，前端就可以一点一点显示内容。
            
            
import os
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/")
async def get():
    """返回聊天页面"""

    file_path = os.path.join(os.path.dirname(__file__), "chat.html")
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            print(f"收到用户消息: {user_message}")
            await ask_stream(user_message,websocket=websocket)
            """
            reply_message = ask(user_message)
            await websocket.send_json({"reply": reply_message})
            """
    except Exception as e:
        print(f"连接关闭: {e}")

import uvicorn

if __name__ == '__main__':

    # 交互式API文档地址：
    # http://127.0.0.1:8000/docs/ 
    # http://127.0.0.1:8000/redoc/

    uvicorn.run(app, host="0.0.0.0", port=8000)