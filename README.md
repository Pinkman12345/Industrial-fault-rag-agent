# 工业设备故障溯源 RAG-Agent 系统

本项目是一个面向工业设备故障诊断与溯源场景的 RAG-Agent Demo，主要用于验证大模型在工业知识库问答、故障原因分析、传感器解释、规则判断和结构化诊断报告生成中的应用流程。

## 当前目标

第一版 MVP 将实现：

- 本地工业设备知识库构建；
- 基于 Chroma 的向量检索；
- 基于 LLM 的 RAG 故障问答；
- 传感器解释工具；
- 异常规则判断工具；
- 故障诊断报告生成工具；
- Streamlit 交互式 Demo。

## 技术栈

- Python
- Streamlit
- LangChain
- Chroma
- OpenAI-compatible API
- Markdown / CSV / JSON

## 项目定位

该项目不是企业级工业运维平台，而是一个用于展示 AI 应用开发、RAG、Agent 工具调用和工业故障诊断场景结合能力的可运行 Demo。