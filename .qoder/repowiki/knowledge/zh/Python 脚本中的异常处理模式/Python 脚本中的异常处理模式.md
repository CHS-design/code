---
kind: error_handling
name: Python 脚本中的异常处理模式
category: error_handling
scope:
    - '**'
source_files:
    - study/研究生学习/8.GoogLeNet/mean_std.py
    - study/研究生学习/9.ResNet/test_me.py
    - study/研究生学习/9.ResNet/train_me.py
---

该仓库为深度学习学习笔记与示例代码集合，不包含 Go、Java 等语言的服务端错误处理体系。在 Python 脚本中，错误处理呈现以下特点：

1. **使用内置异常类型**：所有错误均通过 `raise` 抛出标准 Python 异常，如 `FileNotFoundError`（缺失数据/模型文件）、`ValueError`（类别映射不一致、参数校验失败）。
2. **无自定义错误类**：未发现任何 `class XxxError(Exception)` 形式的自定义异常定义，也未见统一的错误码或错误对象封装。
3. **无全局捕获机制**：脚本未使用 `try/except` 包裹主流程进行统一兜底，仅在个别位置做兼容性兼容（如 PIL 版本差异的 `AttributeError` 捕获），其余均为直接 `raise` 让调用方感知。
4. **无日志框架**：未见 logging 模块或第三方日志库的使用，错误信息以字符串形式随异常抛出，调试依赖终端输出。
5. **无 panic/recover 等价物**：Python 无 panic/recover 概念，且代码中未出现 `sys.exit()` 或 `os._exit()` 等强制退出逻辑。

总体而言，本仓库的错误处理是**零散、基于内置异常的直接抛出**风格，属于教学示例代码的常见写法，不具备跨模块的统一错误处理架构。