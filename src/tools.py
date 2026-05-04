import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd

from config import settings


# =========================
# 1. 传感器知识库
# =========================

SENSOR_KNOWLEDGE = {
    "TJSD": {
        "name": "推进速度",
        "system": "推进系统",
        "meaning": "表示盾构机在单位时间内向前推进的距离，反映设备掘进效率和推进状态。",
        "related_sensors": ["TJL", "DP_ZJ", "DP_SD", "KWC_PRS"],
        "related_faults": ["掌子面阻力增大", "刀具磨损", "推进系统异常", "土仓压力控制不合理"]
    },
    "TJL": {
        "name": "总推进力",
        "system": "推进系统",
        "meaning": "表示推进油缸作用于盾构机的总轴向推力。",
        "related_sensors": ["TJSD", "DP_ZJ", "KWC_PRS"],
        "related_faults": ["推进负载过高", "地层突变", "盾体卡滞", "刀具磨损"]
    },
    "DP_ZJ": {
        "name": "刀盘转矩",
        "system": "刀盘系统",
        "meaning": "表示刀盘旋转过程中受到的阻力矩，是判断开挖负载的重要变量。",
        "related_sensors": ["DP_SD", "TJSD", "TJL", "KWC_PRS"],
        "related_faults": ["刀具磨损", "刀盘卡滞", "掌子面硬度升高", "泥饼形成"]
    },
    "DP_SD": {
        "name": "刀盘转速",
        "system": "刀盘系统",
        "meaning": "表示刀盘单位时间内的旋转速度，需要与推进速度匹配。",
        "related_sensors": ["DP_ZJ", "TJSD", "TJL"],
        "related_faults": ["主驱动负载异常", "刀盘切削状态异常", "刀具磨损"]
    },
    "KWC_PRS": {
        "name": "开挖仓压力",
        "system": "土压/泥水平衡系统",
        "meaning": "表示开挖仓内部压力，用于反映掌子面支护和平衡状态。",
        "related_sensors": ["TJSD", "TJL", "DP_ZJ", "PJGL_FLOW"],
        "related_faults": ["仓压异常", "排浆受阻", "掌子面失衡", "泥水循环异常"]
    },
    "ZJL_LJ": {
        "name": "注浆累计量",
        "system": "注浆系统",
        "meaning": "表示同步注浆过程中累计注入的浆液量。",
        "related_sensors": ["注浆压力", "注浆流量", "推进环号"],
        "related_faults": ["注浆不足", "注浆管路堵塞", "浆液供应异常"]
    },
    "PJGL_FLOW": {
        "name": "排浆管路流量",
        "system": "泥水循环系统",
        "meaning": "表示泥水或渣浆在排浆管路中的流量，反映排浆是否顺畅。",
        "related_sensors": ["KWC_PRS", "DP_ZJ", "TJSD"],
        "related_faults": ["排浆管路堵塞", "泥水循环异常", "开挖仓压力波动"]
    },
    "HBW_YZ_PRS": {
        "name": "主轴承油脂压力",
        "system": "主轴承润滑系统",
        "meaning": "表示主轴承密封或润滑系统中的油脂压力。",
        "related_sensors": ["DP_ZJ", "DP_SD", "主驱动电流"],
        "related_faults": ["主轴承润滑异常", "密封失效", "供脂系统异常"]
    }
}


# =========================
# 2. 工具一：传感器解释工具
# =========================

def sensor_explain_tool(sensor_name: str) -> Dict[str, Any]:
    """
    根据传感器变量名，返回变量含义、所属系统、关联变量和可能故障。

    示例：
    sensor_explain_tool("TJSD")
    """

    sensor_name = sensor_name.strip().upper()

    if sensor_name not in SENSOR_KNOWLEDGE:
        return {
            "success": False,
            "sensor_name": sensor_name,
            "message": f"未找到传感器变量 {sensor_name} 的说明，请检查变量名是否正确。",
            "available_sensors": list(SENSOR_KNOWLEDGE.keys())
        }

    info = SENSOR_KNOWLEDGE[sensor_name]

    return {
        "success": True,
        "sensor_name": sensor_name,
        "chinese_name": info["name"],
        "system": info["system"],
        "meaning": info["meaning"],
        "related_sensors": info["related_sensors"],
        "related_faults": info["related_faults"]
    }


# =========================
# 3. 工具二：异常规则判断工具
# =========================

def anomaly_rule_check_tool(sensor_states: Dict[str, str]) -> Dict[str, Any]:
    """
    根据若干传感器状态，基于简单规则判断异常模式。

    输入示例：
    {
        "TJSD": "下降",
        "DP_ZJ": "升高",
        "TJL": "升高",
        "KWC_PRS": "升高",
        "PJGL_FLOW": "下降"
    }

    注意：
    第一版只做规则判断，不等价于真实故障诊断结论。
    """

    # 统一变量名和值
    states = {
        key.strip().upper(): value.strip()
        for key, value in sensor_states.items()
    }

    matched_rules: List[Dict[str, Any]] = []

    # 规则 1：推进速度下降 + 刀盘转矩升高
    if states.get("TJSD") in ["下降", "降低", "持续下降"] and states.get("DP_ZJ") in ["升高", "上升", "持续升高"]:
        matched_rules.append({
            "rule_id": "R001",
            "rule_name": "推进速度下降且刀盘转矩升高",
            "possible_pattern": "掘进阻力增大 / 刀盘负载异常",
            "reason": "推进速度下降同时刀盘转矩升高，通常说明单位推进过程中的切削阻力增加。",
            "related_faults": ["掌子面阻力增大", "刀具磨损", "刀盘局部卡滞", "泥饼形成"]
        })

    # 规则 2：推进速度下降 + 总推进力升高 + 刀盘转矩升高
    if (
        states.get("TJSD") in ["下降", "降低", "持续下降"]
        and states.get("TJL") in ["升高", "上升", "持续升高"]
        and states.get("DP_ZJ") in ["升高", "上升", "持续升高"]
    ):
        matched_rules.append({
            "rule_id": "R002",
            "rule_name": "推进负载整体升高",
            "possible_pattern": "掌子面阻力增大 / 地层突变 / 盾体受阻",
            "reason": "推进速度下降而总推进力和刀盘转矩同时升高，说明设备为维持掘进需要更高负载。",
            "related_faults": ["地层阻力增大", "盾体卡滞", "刀具磨损", "推进参数不匹配"]
        })

    # 规则 3：仓压升高 + 排浆流量下降
    if states.get("KWC_PRS") in ["升高", "上升", "持续升高"] and states.get("PJGL_FLOW") in ["下降", "降低", "持续下降"]:
        matched_rules.append({
            "rule_id": "R003",
            "rule_name": "仓压升高且排浆流量下降",
            "possible_pattern": "排浆不畅 / 泥水循环异常",
            "reason": "开挖仓压力升高同时排浆流量下降，可能表示排浆管路堵塞或渣土输送不畅。",
            "related_faults": ["排浆管路堵塞", "开挖仓积渣", "泥水循环异常"]
        })

    # 规则 4：注浆累计量不足
    if states.get("ZJL_LJ") in ["不足", "偏低", "下降", "低于预期"]:
        matched_rules.append({
            "rule_id": "R004",
            "rule_name": "注浆累计量不足",
            "possible_pattern": "注浆不足 / 注浆系统异常",
            "reason": "注浆累计量低于预期，可能导致管片背后空隙填充不足。",
            "related_faults": ["注浆泵异常", "注浆管路堵塞", "浆液供应不足"]
        })

    if not matched_rules:
        return {
            "success": True,
            "input_states": states,
            "has_anomaly_pattern": False,
            "matched_rules": [],
            "summary": "未命中当前规则库中的典型异常模式，建议结合历史趋势、阈值范围和更多传感器变量进一步判断。"
        }

    return {
        "success": True,
        "input_states": states,
        "has_anomaly_pattern": True,
        "matched_rules": matched_rules,
        "summary": f"共命中 {len(matched_rules)} 条异常规则，建议结合 RAG 检索结果和时序趋势进一步确认。"
    }


# =========================
# 4. 工具三：趋势分析工具，可选但建议保留
# =========================

def trend_analysis_tool(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """
    读取一小段 CSV 时序数据，计算均值、最大值、最小值和简单趋势。

    默认读取 data/sensor_sample.csv。
    """

    if csv_path is None:
        csv_file = settings.DATA_DIR / "sensor_sample.csv"
    else:
        csv_file = Path(csv_path)

    if not csv_file.exists():
        return {
            "success": False,
            "message": f"CSV 文件不存在: {csv_file}"
        }

    df = pd.read_csv(csv_file)

    if df.empty:
        return {
            "success": False,
            "message": "CSV 文件为空。"
        }

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    results = {}

    for col in numeric_cols:
        values = df[col].dropna()

        if len(values) < 2:
            trend = "数据不足"
        else:
            first_value = values.iloc[0]
            last_value = values.iloc[-1]

            if last_value > first_value:
                trend = "上升"
            elif last_value < first_value:
                trend = "下降"
            else:
                trend = "基本稳定"

        results[col] = {
            "mean": round(float(values.mean()), 4),
            "max": round(float(values.max()), 4),
            "min": round(float(values.min()), 4),
            "first": round(float(values.iloc[0]), 4),
            "last": round(float(values.iloc[-1]), 4),
            "trend": trend
        }

    # 基于趋势的简单辅助判断
    pattern_hints = []

    if results.get("TJSD", {}).get("trend") == "下降" and results.get("DP_ZJ", {}).get("trend") == "上升":
        pattern_hints.append("推进速度下降且刀盘转矩上升，疑似掘进阻力增大或刀盘负载异常。")

    if results.get("TJL", {}).get("trend") == "上升" and results.get("DP_ZJ", {}).get("trend") == "上升":
        pattern_hints.append("总推进力与刀盘转矩同时上升，说明设备负载可能整体增大。")

    if results.get("KWC_PRS", {}).get("trend") == "上升" and results.get("PJGL_FLOW", {}).get("trend") == "下降":
        pattern_hints.append("开挖仓压力上升且排浆流量下降，疑似排浆不畅或泥水循环异常。")

    return {
        "success": True,
        "csv_path": str(csv_file),
        "row_count": len(df),
        "columns": df.columns.tolist(),
        "statistics": results,
        "pattern_hints": pattern_hints
    }


# =========================
# 5. 工具四：故障报告生成工具
# =========================

def fault_report_tool(
    question: str,
    rag_answer: str,
    rule_check_result: Optional[Dict[str, Any]] = None,
    trend_result: Optional[Dict[str, Any]] = None,
    output_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    根据 RAG 回答、规则判断结果、趋势分析结果，生成 Markdown 故障诊断报告。

    输出位置：
    outputs/report_xxx.md
    """

    settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"fault_report_{timestamp}.md"

    output_path = settings.OUTPUTS_DIR / output_filename

    report_lines = []

    report_lines.append("# 工业设备故障诊断报告")
    report_lines.append("")
    report_lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"- 用户问题：{question}")
    report_lines.append("")

    report_lines.append("## 一、RAG 诊断回答")
    report_lines.append("")
    report_lines.append(rag_answer.strip())
    report_lines.append("")

    if rule_check_result is not None:
        report_lines.append("## 二、规则工具判断结果")
        report_lines.append("")
        report_lines.append(f"- 是否命中异常模式：{rule_check_result.get('has_anomaly_pattern')}")
        report_lines.append(f"- 判断摘要：{rule_check_result.get('summary')}")
        report_lines.append("")

        matched_rules = rule_check_result.get("matched_rules", [])
        if matched_rules:
            report_lines.append("### 命中规则")
            report_lines.append("")
            for rule in matched_rules:
                report_lines.append(f"#### {rule.get('rule_id')} {rule.get('rule_name')}")
                report_lines.append("")
                report_lines.append(f"- 可能模式：{rule.get('possible_pattern')}")
                report_lines.append(f"- 判断依据：{rule.get('reason')}")
                report_lines.append(f"- 关联故障：{', '.join(rule.get('related_faults', []))}")
                report_lines.append("")

    if trend_result is not None:
        report_lines.append("## 三、趋势分析结果")
        report_lines.append("")
        report_lines.append(f"- 数据文件：{trend_result.get('csv_path')}")
        report_lines.append(f"- 数据行数：{trend_result.get('row_count')}")
        report_lines.append("")

        statistics = trend_result.get("statistics", {})
        if statistics:
            report_lines.append("| 变量 | 均值 | 最小值 | 最大值 | 首值 | 末值 | 趋势 |")
            report_lines.append("|---|---:|---:|---:|---:|---:|---|")

            for col, stat in statistics.items():
                report_lines.append(
                    f"| {col} | {stat.get('mean')} | {stat.get('min')} | {stat.get('max')} | "
                    f"{stat.get('first')} | {stat.get('last')} | {stat.get('trend')} |"
                )

            report_lines.append("")

        pattern_hints = trend_result.get("pattern_hints", [])
        if pattern_hints:
            report_lines.append("### 趋势辅助判断")
            report_lines.append("")
            for hint in pattern_hints:
                report_lines.append(f"- {hint}")
            report_lines.append("")

    report_lines.append("## 四、报告说明")
    report_lines.append("")
    report_lines.append(
        "本报告由 RAG 检索结果、大模型结构化回答、规则判断工具和可选趋势分析工具共同生成。"
        "当前结果用于故障辅助分析与 Demo 展示，不应替代现场工程诊断结论。"
    )
    report_lines.append("")

    output_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "success": True,
        "report_path": str(output_path),
        "message": f"故障诊断报告已生成: {output_path}"
    }


# =========================
# 6. 本文件单独测试
# =========================

if __name__ == "__main__":
    print("=" * 80)
    print("测试 sensor_explain_tool")
    print("=" * 80)
    print(json.dumps(sensor_explain_tool("TJSD"), ensure_ascii=False, indent=2))

    print("\n" + "=" * 80)
    print("测试 anomaly_rule_check_tool")
    print("=" * 80)

    sample_states = {
        "TJSD": "下降",
        "DP_ZJ": "升高",
        "TJL": "升高",
        "KWC_PRS": "升高",
        "PJGL_FLOW": "下降"
    }

    rule_result = anomaly_rule_check_tool(sample_states)
    print(json.dumps(rule_result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 80)
    print("测试 trend_analysis_tool")
    print("=" * 80)

    trend_result = trend_analysis_tool()
    print(json.dumps(trend_result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 80)
    print("测试 fault_report_tool")
    print("=" * 80)

    mock_rag_answer = """
## 1. 故障现象概括
推进速度下降，同时刀盘转矩和总推进力升高，说明设备掘进负载可能增大。

## 2. 可能原因分析
可能与掌子面阻力增大、刀具磨损、排浆不畅或仓压异常有关。
"""

    report_result = fault_report_tool(
        question="推进速度下降，同时刀盘转矩升高，可能是什么原因？",
        rag_answer=mock_rag_answer,
        rule_check_result=rule_result,
        trend_result=trend_result,
        output_filename="report_example.md"
    )

    print(json.dumps(report_result, ensure_ascii=False, indent=2))