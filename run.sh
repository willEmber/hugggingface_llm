#!/bin/bash

# 设置所有需要的环境变量
export HF_ENDPOINT="https://hf-mirror.com"
# 如果未来有其他变量，也可以加在这里
# export ANOTHER_VAR="some_value"

# 切换到脚本所在的目录 (这是一个好习惯，可以确保脚本从任何地方都能正确运行)
cd "$(dirname "$0")"

# 运行你的 Python 应用
python main.py