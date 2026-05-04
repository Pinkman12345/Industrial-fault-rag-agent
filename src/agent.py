from typing import Dict, Any, List

from rag_chain import rag_answer
from tools import (
    sensor_explain_tool,
    anomaly_rule_check_tool,
    trend_analysis_tool,
    fault_report_tool
)


def infer_sensor_states_from_question(question: str) -> Dict[str, str]:
    """
    根据用户输入的中文故障现象，做一个简单的传感器状态抽取。

    第一版采用关键词规则，不依赖大模型。
    后续可以升级为：
    1. LLM 结构化抽取；
    2. JSON 输出解析；
    3. 与真实传感器阈值判断结合。

    示例：
    输入：推进速度下降，同时刀盘转矩升高
    输出：{"TJSD": "下降", "DP_ZJ": "升高"}
    """

    states: Dict[str, str] = {}

    # 推进速度
    if "推进速度" in question:
        if any(word in question for word in ["下降", "降低", "变慢", "减小", "持续下降"]):
            states["TJSD"] = "下降"
        elif any(word in question for word in ["升高", "上升", "变快", "增大"]):
            states["TJSD"] = "升高"

    # 总推进力
    if "总推进力" in question or "推进力" in question:
        if any(word in question for word in ["升高", "上升", "增大", "过高", "持续升高"]):
            states["TJL"] = "升高"
        elif any(word in question for word in ["下降", "降低", "减小"]):
            states["TJL"] = "下降"

    # 刀盘转矩
    if "刀盘转矩" in question or "转矩" in question or "扭矩" in question:
        if any(word in question for word in ["升高", "上升", "增大", "过高", "持续升高"]):
            states["DP_ZJ"] = "升高"
        elif any(word in question for word in ["下降", "降低", "减小"]):
            states["DP_ZJ"] = "下降"

    # 刀盘转速
    if "刀盘转速" in question or "转速" in question:
        if any(word in question for word in ["升高", "上升", "增大", "变快"]):
            states["DP_SD"] = "升高"
        elif any(word in question for word in ["下降", "降低", "减小", "变慢"]):
            states["DP_SD"] = "下降"

    # 开挖仓压力
    if "开挖仓压力" in question or "仓压" in question or "土仓压力" in question:
        if any(word in question for word in ["升高", "上升", "增大", "过高", "偏高", "持续升高"]):
            states["KWC_PRS"] = "升高"
        elif any(word in question for word in ["下降", "降低", "偏低", "减小"]):
            states["KWC_PRS"] = "下降"
        elif any(word in question for word in ["波动", "不稳定", "震荡"]):
            states["KWC_PRS"] = "波动"

    # 排浆流量
    if "排浆流量" in question or "排浆管路流量" in question or "排浆" in question:
        if any(word in question for word in ["下降", "降低", "减小", "不足", "变小"]):
            states["PJGL_FLOW"] = "下降"
        elif any(word in question for word in ["升高", "上升", "增大"]):
            states["PJGL_FLOW"] = "升高"

    # 注浆累计量
    if "注浆累计量" in question or "注浆量" in question or "注浆" in question:
        if any(word in question for word in ["不足", "偏低", "下降", "降低", "低于预期"]):
            states["ZJL_LJ"] = "不足"
        elif any(word in question for word in ["升高", "增加", "过高"]):
            states["ZJL_LJ"] = "升高"

    return states


def explain_related_sensors(sensor_states: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    对已经识别出的传感器变量调用 sensor_explain_tool。
    """

    explanations = []

    for sensor_name in sensor_states.keys():
        result = sensor_explain_tool(sensor_name)
        explanations.append(result)

    return explanations


def run_fault_diagnosis_agent(question: str, use_trend_analysis: bool = True) -> Dict[str, Any]:
    """
    工业设备故障诊断 Agent 主流程。

    工作流：
    1. RAG 问答；
    2. 从用户问题中抽取传感器状态；
    3. 调用传感器解释工具；
    4. 调用异常规则判断工具；
    5. 可选调用趋势分析工具；
    6. 调用报告生成工具；
    7. 返回完整结果。
    """

    print("步骤 1：执行 RAG 检索问答...")
    rag_result = rag_answer(question, top_k=4)
    rag_text = rag_result["answer"]

    print("步骤 2：从问题中抽取传感器状态...")
    sensor_states = infer_sensor_states_from_question(question)

    print("识别到的传感器状态：", sensor_states)

    print("步骤 3：调用传感器解释工具...")
    sensor_explanations = explain_related_sensors(sensor_states)

    print("步骤 4：调用异常规则判断工具...")
    rule_check_result = anomaly_rule_check_tool(sensor_states)

    trend_result = None
    if use_trend_analysis:
        print("步骤 5：调用趋势分析工具...")
        trend_result = trend_analysis_tool()

    print("步骤 6：生成故障诊断报告...")
    report_result = fault_report_tool(
        question=question,
        rag_answer=rag_text,
        rule_check_result=rule_check_result,
        trend_result=trend_result
    )

    final_result = {
        "question": question,
        "rag_answer": rag_text,
        "sensor_states": sensor_states,
        "sensor_explanations": sensor_explanations,
        "rule_check_result": rule_check_result,
        "trend_result": trend_result,
        "report_result": report_result,
        "retrieved_docs": rag_result["retrieved_docs"]
    }

    return final_result


def print_agent_result(result: Dict[str, Any]) -> None:
    """
    终端友好打印 Agent 运行结果。
    """

    print("\n" + "=" * 80)
    print("Agent 诊断完成")
    print("=" * 80)

    print("\n【用户问题】")
    print(result["question"])

    print("\n【识别到的传感器状态】")
    print(result["sensor_states"])

    print("\n【规则判断摘要】")
    print(result["rule_check_result"].get("summary"))

    matched_rules = result["rule_check_result"].get("matched_rules", [])
    if matched_rules:
        print("\n【命中规则】")
        for rule in matched_rules:
            print(f"- {rule.get('rule_id')} {rule.get('rule_name')}：{rule.get('possible_pattern')}")

    if result.get("trend_result"):
        print("\n【趋势辅助判断】")
        pattern_hints = result["trend_result"].get("pattern_hints", [])
        if pattern_hints:
            for hint in pattern_hints:
                print(f"- {hint}")
        else:
            print("未发现明显趋势提示。")

    print("\n【RAG 诊断回答】")
    print(result["rag_answer"])

    print("\n【报告生成结果】")
    print(result["report_result"].get("message"))


if __name__ == "__main__":
    test_question = "推进速度下降，同时刀盘转矩升高，总推进力也升高，可能是什么原因？"

    result = run_fault_diagnosis_agent(
        question=test_question,
        use_trend_analysis=True
    )

    print_agent_result(result)