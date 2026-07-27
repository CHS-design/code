# 项目复盘

## 2026-07-22：PDF 复合图不能只提取内嵌位图

- PDF 中的论文插图可能由位图、矢量线条和文字标签共同组成。只提取 `image` 对象会得到残缺图片，即使 DOCX 中已有对应图号，也不能据此判断图片完整。
- PDF 转译或整理后，应同时核对图号数量和页面渲染效果。遇到复合图时，优先从原 PDF 页面按图框高清裁切，再插入 DOCX。
- 修订带有学习标注的 DOCX 时，应只替换目标媒体和对应尺寸节点，并验证包内其他文件内容未变化，避免破坏字体颜色、高亮等现有标注。

## 2026-07-25：Windows 下的 PPT 模板工作流

- `presentations` 的 `artifact-tool` 辅助脚本会根据 `HOME` 推断桌面运行时位置；在本机 PowerShell 中执行前应显式设置 `$env:HOME='C:\Users\admin'`，否则会误查工作目录下的 `.cache` 并报缺少 `@oai/artifact-tool`。
- 模板检查脚本依赖 `unzip`；此 Windows 环境没有该命令时，保留 `artifact-tool` 导入/导出流程，并用系统 `tar` 的只读列出和提取能力完成检查，不能直接修改 PPTX OOXML。

## 2026-07-26：Windows 下 DOCX 渲染需要使用独立 LibreOffice 配置目录

- 本机直接通过通用 `render_docx.py` 调用 LibreOffice 可能超时。对 DOCX 做视觉核对时，可改用 `soffice.com --headless --convert-to pdf`，并通过 `-env:UserInstallation=file:///...` 指定一个独立、可写的配置目录。
- `soffice.com` 的控制台输出可能不是 UTF-8；自动化调用应以字节方式捕获输出。PDF 转 PNG 可使用已安装的 `pypdfium2`，避免依赖未加入 `PATH` 的 Poppler。
