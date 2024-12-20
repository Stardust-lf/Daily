#!/bin/bash

# 编译命令
g++ -g -Wall -Werror -O0 find_conflict.cc -o find_conflict

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "Build failed: Compilation error in find_conflict.cc"
    exit 1
fi

# 运行生成的可执行文件
./find_conflict
if [ $? -ne 0 ]; then
    echo "Execution failed: Error running find_conflict"
    exit 1
fi

echo "Build and execution successful."
