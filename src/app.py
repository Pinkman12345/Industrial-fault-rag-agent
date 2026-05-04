import sys
from pathlib import Path
from typing import Dict, Any

import streamlit as st

# 确保可以从 src 目录导入本项目模块
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from agent import run_fault_diagnosis_agent
from tools import sensor_explain_tool
from vector_store import build_vector_store, search_similar_documents


# =========================
# 页面基础配置
# =========================

st.set_page_config(
    page_title="工业设备故障溯源 RAG-Agent 系统",
    page_icon="🛠️",
    layout="wide"
)


st.title("🛠️ 工业设备故障溯源 RAG-Agent 系统")

st.markdown(
    """
本系统是一个面向工业设备/盾构机故障诊断场景的 **RAG-Agent Demo**，支持：

- 基于本地知识库的故障问答；
- 传感器变量解释；
- 异常规则判断；
- CSV 时序数据趋势分析；
- Markdown 结构化诊断报告生成。

当前项目用于展示 **LLM 应用开发、RAG、Agent 工具调用、Prompt Engineering 与工业故障诊断场景结合能力**。
"""
)


# =========================
# 侧边栏
# =========================

with st.sidebar:
    st.header("⚙️ 系统控制")

    top_k = st.slider(
        "知识库检索 Top-K",
        min_value=2,
        max_value=8,
        value=4,
        step=1
    )

    use_trend_analysis = st.checkbox(
        "启用 CSV 趋势分析工具",
        value=True
    )

    st.divider()

    st.subheader("📚 向量库操作")

    if st.button("重新构建向量库"):
        with st.spinner("正在重新构建 Chroma 向量库..."):
            try:
                build_vector_store()
                st.success("向量库构建完成")
            except Exception as e:
                st.error(f"向量库构建失败：{e}")

    st.divider()

    st.subheader("💡 示例问题")

    example_questions = [
        "推进速度下降，同时刀盘转矩升高，总推进力也升高，可能是什么原因？",
        "开挖仓压力升高，同时排浆流量下降，应该检查什么？",
        "注浆量不足可能会导致哪些问题？",
        "刀盘转矩持续升高，但是推进速度下降，是否可能是刀具磨损？"
    ]

    selected_example = st.selectbox(
        "选择一个示例问题",
        example_questions
    )

    st.caption("选择后可以复制到主输入框中进行测试。")


# =========================
# 主输入区
# =========================

st.subheader("🔍 故障现象输入")

default_question = selected_example

question = st.text_area(
    "请输入故障现象或诊断问题：",
    value=default_question,
    height=120,
    placeholder="例如：推进速度下降，同时刀盘转矩升高，总推进力也升高，可能是什么原因？"
)

col_run, col_clear = st.columns([1, 1])

with col_run:
    run_button = st.button("开始诊断", type="primary")

with col_clear:
    clear_button = st.button("清空输入")


if clear_button:
    st.rerun()


# =========================
# 诊断执行
# =========================

if run_button:
    if not question.strip():
        st.warning("请先输入故障现象或诊断问题。")
    else:
        with st.spinner("Agent 正在执行：RAG 检索、工具调用与报告生成..."):
            try:
                result = run_fault_diagnosis_agent(
                    question=question.strip(),
                    use_trend_analysis=use_trend_analysis
                )
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"诊断过程中发生错误：{e}")


# =========================
# 结果展示函数
# =========================

def show_sensor_states(sensor_states: Dict[str, str]) -> None:
    st.markdown("### 1. 识别到的传感器状态")

    if not sensor_states:
        st.info("未从问题中识别到明确的传感器状态。")
        return

    st.json(sensor_states)


def show_rule_result(rule_result: Dict[str, Any]) -> None:
    st.markdown("### 2. 异常规则判断结果")

    st.write(rule_result.get("summary", "无规则判断摘要。"))

    matched_rules = rule_result.get("matched_rules", [])

    if not matched_rules:
        st.info("未命中当前规则库中的典型异常模式。")
        return

    for rule in matched_rules:
        with st.expander(f"{rule.get('rule_id')} - {rule.get('rule_name')}", expanded=True):
            st.markdown(f"**可能模式：** {rule.get('possible_pattern')}")
            st.markdown(f"**判断依据：** {rule.get('reason')}")

            related_faults = rule.get("related_faults", [])
            if related_faults:
                st.markdown("**关联故障：**")
                for fault in related_faults:
                    st.markdown(f"- {fault}")


def show_trend_result(trend_result: Dict[str, Any]) -> None:
    st.markdown("### 3. CSV 趋势分析结果")

    if not trend_result:
        st.info("未启用趋势分析工具。")
        return

    if not trend_result.get("success"):
        st.warning(trend_result.get("message", "趋势分析失败。"))
        return

    st.markdown(f"**数据文件：** `{trend_result.get('csv_path')}`")
    st.markdown(f"**数据行数：** {trend_result.get('row_count')}")

    statistics = trend_result.get("statistics", {})

    if statistics:
        table_data = []
        for col, stat in statistics.items():
            table_data.append(
                {
                    "变量": col,
                    "均值": stat.get("mean"),
                    "最小值": stat.get("min"),
                    "最大值": stat.get("max"),
                    "首值": stat.get("first"),
                    "末值": stat.get("last"),
                    "趋势": stat.get("trend")
                }
            )

        st.dataframe(table_data, use_container_width=True)

    pattern_hints = trend_result.get("pattern_hints", [])

    if pattern_hints:
        st.markdown("**趋势辅助判断：**")
        for hint in pattern_hints:
            st.markdown(f"- {hint}")
    else:
        st.info("未发现明显趋势提示。")


def show_rag_answer(rag_answer: str) -> None:
    st.markdown("### 4. RAG 诊断回答")
    st.markdown(rag_answer)


def show_sensor_explanations(sensor_explanations: list) -> None:
    st.markdown("### 5. 相关传感器解释")

    if not sensor_explanations:
        st.info("暂无传感器解释结果。")
        return

    for item in sensor_explanations:
        if not item.get("success"):
            continue

        title = f"{item.get('sensor_name')} - {item.get('chinese_name')}"
        with st.expander(title, expanded=False):
            st.markdown(f"**所属系统：** {item.get('system')}")
            st.markdown(f"**变量含义：** {item.get('meaning')}")

            related_sensors = item.get("related_sensors", [])
            if related_sensors:
                st.markdown("**关联变量：**")
                st.markdown(", ".join(related_sensors))

            related_faults = item.get("related_faults", [])
            if related_faults:
                st.markdown("**可能关联故障：**")
                for fault in related_faults:
                    st.markdown(f"- {fault}")


def show_retrieved_docs(retrieved_docs: list) -> None:
    st.markdown("### 6. 检索到的知识片段")

    if not retrieved_docs:
        st.info("暂无检索结果。")
        return

    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")

        with st.expander(f"片段 {i} | 来源：{source} | 编号：{chunk_id}", expanded=False):
            st.text(doc.page_content[:1200])


def show_report_result(report_result: Dict[str, Any]) -> None:
    st.markdown("### 7. 诊断报告生成结果")

    if not report_result:
        st.info("暂无报告生成结果。")
        return

    if report_result.get("success"):
        st.success(report_result.get("message"))
        st.code(report_result.get("report_path"))
    else:
        st.warning("报告生成失败。")


# =========================
# 展示最近一次结果
# =========================

if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    st.divider()
    st.subheader("📋 Agent 诊断结果")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["诊断结论", "工具结果", "知识依据", "报告"]
    )

    with tab1:
        show_rag_answer(result.get("rag_answer", ""))

    with tab2:
        show_sensor_states(result.get("sensor_states", {}))
        show_rule_result(result.get("rule_check_result", {}))
        show_trend_result(result.get("trend_result"))

    with tab3:
        show_sensor_explanations(result.get("sensor_explanations", []))
        show_retrieved_docs(result.get("retrieved_docs", []))

    with tab4:
        show_report_result(result.get("report_result", {}))

else:
    st.info("请输入故障现象并点击“开始诊断”。")


# =========================
# 底部说明
# =========================

st.divider()

st.caption(
    "说明：本系统为 RAG-Agent Demo，诊断结果仅用于辅助分析和项目展示，不能替代真实工程现场诊断结论。"
)