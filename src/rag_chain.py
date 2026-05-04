from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from vector_store import search_similar_documents


SYSTEM_PROMPT = """
你是一名工业设备故障诊断与溯源助手，面向盾构机、掘进装备、复杂工业设备等场景。

你的任务是：
1. 基于给定的知识库检索内容回答问题；
2. 分析故障现象、可能原因、相关传感器、建议检查项和处理建议；
3. 尽量保持结构化、工程化、可复核；
4. 如果知识库中没有足够依据，要明确说明“不确定”或“需要进一步数据验证”；
5. 不要编造不存在的传感器、故障案例或处理措施。

回答风格要求：
- 使用中文；
- 面向工程诊断场景；
- 不要空泛安慰；
- 不要只给概念解释；
- 优先结合检索到的知识片段进行分析。
"""


RAG_PROMPT_TEMPLATE = """
请基于下面的【知识库检索内容】，回答用户的故障诊断问题。

【用户问题】
{question}

【知识库检索内容】
{context}

【输出要求】
请严格按照下面结构回答：

## 1. 故障现象概括
用 1-3 句话概括用户描述的异常现象。

## 2. 可能原因分析
按可能性从高到低列出 3-5 条原因。
每条原因需要说明依据来自哪些现象或知识库内容。

## 3. 相关传感器
列出与该问题相关的传感器变量，并说明其含义和诊断作用。

## 4. 建议检查项
给出现场或数据侧建议检查的项目。

## 5. 处理建议
给出初步处理建议，注意不要写成绝对结论。

## 6. 不确定性与补充数据需求
说明当前判断还需要哪些数据进一步验证。

## 7. 参考知识片段
列出本次回答主要参考了哪些知识库片段，格式为：
- 来源文件：xxx，片段编号：xxx
"""


def get_llm() -> ChatOpenAI:
    """
    初始化 OpenAI 兼容聊天模型。

    通过 .env 中的配置，可以切换 DeepSeek / Qwen / OpenAI 等模型。
    """

    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_api_key_here":
        raise ValueError(
            "未检测到有效的 OPENAI_API_KEY。请在项目根目录 .env 文件中填写真实 API Key。"
        )

    llm = ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=0.2
    )

    return llm


def format_retrieved_context(docs: List[Document]) -> str:
    """
    将检索到的 Document 片段拼接成可放入 Prompt 的上下文。
    """

    context_parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")

        part = f"""
[片段 {i}]
来源文件：{source}
片段编号：{chunk_id}
内容：
{doc.page_content}
"""
        context_parts.append(part.strip())

    return "\n\n".join(context_parts)


def rag_answer(question: str, top_k: int = 4) -> Dict[str, Any]:
    """
    RAG 故障问答主函数。

    流程：
    1. 根据用户问题检索相关文档；
    2. 拼接上下文；
    3. 构造 Prompt；
    4. 调用大模型；
    5. 返回回答和检索片段。
    """

    retrieved_docs = search_similar_documents(question, top_k=top_k)
    context = format_retrieved_context(retrieved_docs)

    user_prompt = RAG_PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )

    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    return {
        "question": question,
        "answer": response.content,
        "retrieved_docs": retrieved_docs
    }


def print_rag_result(result: Dict[str, Any]) -> None:
    """
    以便于终端查看的形式打印 RAG 结果。
    """

    print("=" * 80)
    print("用户问题：")
    print(result["question"])
    print("=" * 80)

    print("\n模型回答：")
    print(result["answer"])

    print("\n" + "=" * 80)
    print("检索到的知识片段：")
    print("=" * 80)

    for i, doc in enumerate(result["retrieved_docs"], start=1):
        print(f"\n片段 {i}")
        print("来源文件:", doc.metadata.get("source"))
        print("片段编号:", doc.metadata.get("chunk_id"))
        print("内容预览:")
        print(doc.page_content[:300])
        print("-" * 80)


if __name__ == "__main__":
    test_question = "推进速度下降，同时刀盘转矩升高，可能是什么原因？"

    result = rag_answer(test_question, top_k=4)
    print_rag_result(result)