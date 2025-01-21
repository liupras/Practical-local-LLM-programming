@echo off
echo start translation service

REM 修改代码页，防止在windows中执行时出现中文乱码
chcp 65001

REM 切换到当前批处理文件所在的目录
cd /d %~dp0

echo 激活虚拟环境...
call ..\..\.venv\Scripts\activate

echo 启动大语言模型服务...
python api.py

pause