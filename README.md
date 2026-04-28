# LLM Inference Notes

一个面向阅读的 LLM 推理笔记静态站，包含章节目录、全文搜索、字号切换、深浅色模式和 MathJax 公式渲染。

## 在线访问

如果仓库已开启 GitHub Pages，可以直接访问：

https://li199959.github.io/llm-inference-notes/

GitHub Pages 设置建议：

1. 打开仓库的 `Settings` -> `Pages`。
2. `Build and deployment` 选择 `Deploy from a branch`。
3. Branch 选择 `main`，目录选择 `/ (root)`。
4. 保存后等待 GitHub 生成页面。

## 本地预览

这个项目是纯静态页面，不需要安装依赖。进入项目目录后启动一个本地静态服务：

```powershell
python -m http.server 3000 --bind 127.0.0.1
```

然后打开：

http://127.0.0.1:3000/

## 项目结构

- `index.html`：阅读器入口页面。
- `reader.css`：页面样式。
- `reader.js`：章节加载、搜索、目录、主题切换和公式渲染逻辑。
- `translated_md/`：中文 Markdown 章节内容。
- `source_text/`：原始文本资料。

## 注意

原始 PDF 文件 `llm.pdf` 不包含在仓库中，已通过 `.gitignore` 排除。
