---
kind: configuration_system
name: Git 忽略规则与本地配置排除策略
category: configuration_system
scope:
    - '**'
source_files:
    - .gitignore
---

本仓库采用单文件 `.gitignore` 集中管理版本控制忽略规则，未使用任何运行时配置加载框架（如 `config/`、`.env` 解析器、YAML/TOML 配置中心）。该 `.gitignore` 是仓库中唯一与“配置系统”相关的代码，其职责是**排除本地环境噪声与敏感信息**，而非提供应用级配置管理能力。具体策略如下：

1. **操作系统与编辑器产物**：忽略 `.DS_Store`、`Thumbs.db`、`.idea/`、`*.iml`；保留 VS Code 共享扩展配置 `!.vscode/extensions.json`。
2. **本地环境与密钥**：显式忽略 `.env`、`.env.*`、`*.local`，防止凭据入库。
3. **Python 生态缓存与虚拟环境**：`__pycache__/`、`*.py[cod]`、`.pytest_cache/`、`.mypy_cache/`、`.ruff_cache/`、`.venv/`、`venv/`、`env/`。
4. **Jupyter Notebook 检查点**：`.ipynb_checkpoints/`。
5. **构建/渲染输出**：`_build/` 及其递归匹配 `**/_build/`。
6. **机器学习数据与模型权重**：`data/`、`**/data/`、`data_me/`，以及常见二进制格式 `*.pth`、`*.pt`、`*.ckpt`、`*.onnx`、`*.h5`、`*.npy`、`*.npz`、`*.pkl`、`*.joblib`。
7. **归档与压缩数据集**：`*.zip`、`*.rar`、`*.7z`、`*.tar`、`*.tar.gz`、`*.tgz`、`*.gz`。
8. **日志与临时文件**：`*.log`、`*.tmp`、`*.temp`、`~$*`。

开发者约定：
- 所有运行期配置应通过环境变量注入，不得将 `.env` 提交到仓库。
- 训练产生的模型权重与原始数据统一放入 `data/` 或 `data_me/`，由 `.gitignore` 自动排除。
- 如需在团队内共享 VS Code 设置，仅保留 `extensions.json`，其余工作区配置各自本地化。