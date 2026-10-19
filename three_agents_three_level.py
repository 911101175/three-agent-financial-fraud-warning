#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于结构化工作底稿的三智能体协同分析系统
=========================================

实现“指控—辩护—裁决”三智能体顺序博弈流程，用于识别上市公司
总体财务舞弊风险，输出统一的总体风险等级。

设计严格遵循《基于结构化工作底稿的三智能体构建思路》：
  指控智能体(prosecution)  -> 发现风险，避免漏报
  辩护智能体(defense)      -> 解释合理性，降低误报
  裁决智能体(verdict)      -> 综合裁决，形成最终风险等级与核查建议

事实基础：workpaper_generator.py 生成的结构化工作底稿（JSON / dict）。
三智能体不重新计算指标，只基于底稿中的结构化证据进行推理。

依赖：
    pip install openai>=1.0
    （DeepSeek 兼容 OpenAI SDK 接口）

用法：
    # 1) 已有底稿 JSON 文件
    python three_agents.py --workpaper workpaper_000651_2023.json --out result.json

    # 2) 直接从底稿生成器生成底稿再分析（需要数据 CSV + 模型文件在同目录）
    python three_agents.py --stock_code 000651 --year 2023 --out result.json
    python three_agents.py --stock_code 002740 --year 2020 --out result.json

    # 设置 API Key（二选一）
    export DEEPSEEK_API_KEY="YOUR_API_KEY"
    # 或运行时 --api_key YOUR_API_KEY
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ── DeepSeek 配置 ──────────────────────────────────────────────
BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"     # 也可换 deepseek-v4-pro
DEFAULT_TEMPERATURE = 0.2


# ════════════════════════════════════════════════════════════════
#  一、底稿切片：根据不同智能体的分析重点，从完整底稿中提取相关模块
# ════════════════════════════════════════════════════════════════
def _module_value(workpaper: Dict[str, Any], legacy_key: str, modern_key: str,
                  default: Any = None) -> Any:
    """兼容平铺底稿和 workpaper_generator 当前 A-G 底稿。"""
    value = workpaper.get(legacy_key)
    if value is not None:
        return value
    value = workpaper.get(modern_key)
    return default if value is None else value


_PRIVATE_KEY_PARTS = (
    "stock_code", "security_code", "company_name", "companyname",
    "公司名称", "股票代码", "证券代码", "统一社会信用代码",
)


def _is_private_key(key: Any) -> bool:
    text = str(key).lower().replace("-", "_").replace(" ", "")
    return any(part in text for part in _PRIVATE_KEY_PARTS)


def _clean_private(value: Any) -> Any:
    """递归移除企业身份字段；其余底稿字段完整保留。"""
    if isinstance(value, dict):
        return {
            str(key): _clean_private(item)
            for key, item in value.items()
            if not _is_private_key(key)
        }
    if isinstance(value, list):
        return [_clean_private(item) for item in value]
    return value


def _full_agent_evidence(workpaper: Dict[str, Any]) -> Dict[str, Any]:
    """三位智能体共享同一份完整、去身份信息的底稿。"""
    return _clean_private(workpaper)


def _public_basic_info(workpaper: Dict[str, Any]) -> Dict[str, Any]:
    info = _module_value(workpaper, "basic_info", "A_sample_basic_info", {}) or {}
    # A 模块会保留内部身份字段供留档，但不能进入模型提示词。
    return {
        str(key): value for key, value in info.items()
        if not str(key).startswith("_")
        and str(key).lower() not in {"stock_code", "company_name", "security_code"}
    }


def _rule_details(workpaper: Dict[str, Any]) -> List[Dict[str, Any]]:
    block = workpaper.get("D_rule_hit_results") or workpaper.get("D_rule_results") or {}
    if isinstance(block, dict):
        rules = block.get("all_rules_detail") or block.get("rules") or []
        if isinstance(rules, list):
            return [item for item in rules if isinstance(item, dict)]
    rules = workpaper.get("rules") or []
    return [item for item in rules if isinstance(item, dict)] if isinstance(rules, list) else []


def _workpaper_risk_summary(workpaper: Dict[str, Any]) -> Dict[str, Any]:
    existing = workpaper.get("risk_summary")
    if isinstance(existing, dict):
        return dict(existing)
    rules = _rule_details(workpaper)
    hits = [rule for rule in rules if rule.get("rule_hit") == 1]
    return {"hit_count": len(hits), "risk_flags": [rule.get("rule_id") for rule in hits]}


def _engine_status_from_workpaper(workpaper: Dict[str, Any]) -> Dict[str, Any]:
    existing = workpaper.get("engine_status")
    if isinstance(existing, dict):
        return existing
    rules = _rule_details(workpaper)
    quality = workpaper.get("F_data_quality") or {}
    unusable = quality.get("unusable_rules") or []
    unusable_ids = {str(item.get("rule_id")) for item in unusable if isinstance(item, dict)}
    return {
        "total_rules": len(rules),
        "normal_calculated": len(rules) - len(unusable_ids),
        "details": {
            str(rule.get("rule_id")): {
                "status": rule.get("rule_status") if str(rule.get("rule_id")) in unusable_ids else "正常",
                "name": rule.get("rule_name", ""),
            }
            for rule in rules
        },
    }


def slice_for_prosecution(wp: Dict[str, Any]) -> Dict[str, Any]:
    """指控智能体读取完整底稿和 ML 结果，但不读取身份字段。"""
    return _full_agent_evidence(wp)


def slice_for_defense(wp: Dict[str, Any]) -> Dict[str, Any]:
    """辩护智能体读取与指控智能体相同的完整底稿和 ML 结果。"""
    return _full_agent_evidence(wp)


def slice_for_verdict(wp: Dict[str, Any]) -> Dict[str, Any]:
    """裁决智能体读取与前两位相同的完整底稿和 ML 结果。"""
    return _full_agent_evidence(wp)


def _engine_brief(engine_status: Dict[str, Any]) -> Dict[str, Any]:
    """提取引擎状态中“不可作为主要依据”的规则清单。"""
    details = (engine_status or {}).get("details", {}) or {}
    unusable = []
    for code, d in details.items():
        status = d.get("status", "")
        if status in ("停用", "未实现", "计算异常", "数据不足", "calculation_error", "data_insufficient"):
            unusable.append({"rule_code": code, "status": status, "name": d.get("name", "")})
    return {
        "total_rules": engine_status.get("total_rules"),
        "normal_calculated": engine_status.get("normal_calculated"),
        "unusable_rules": unusable,
        "note": "unusable_rules 中的规则因停用/未实现/计算异常/数据不足，不得作为主要指控或裁决依据。",
    }


# ════════════════════════════════════════════════════════════════
#  二、LLM 客户端封装（DeepSeek，OpenAI 兼容）
# ════════════════════════════════════════════════════════════════
class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = BASE_URL,
                 model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE):
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("[ERROR] 缺少 openai 库，请先运行: pip install openai")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def chat_json(self, system_prompt: str, user_prompt: str,
                  max_retries: int = 3) -> Dict[str, Any]:
        """调用模型并强制返回 JSON（解析失败自动重试）。"""
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = resp.choices[0].message.content
                return _safe_parse_json(content)
            except Exception as e:
                last_err = e
                print(f"[WARN] 第 {attempt} 次调用失败: {e}")
                time.sleep(1.5 * attempt)
        raise RuntimeError(f"LLM 调用多次失败: {last_err}")


def _safe_parse_json(text: str) -> Dict[str, Any]:
    """从模型输出中稳健地解析 JSON。"""
    if text is None:
        raise ValueError("模型返回为空")
    text = text.strip()
    # 去掉可能的 ```json 围栏
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 退而求其次：截取第一个 { 到最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


# ════════════════════════════════════════════════════════════════
#  三、提示词（System Prompt）：固化角色、规则映射与固定证据约束
# ════════════════════════════════════════════════════════════════
RISK_LEVELS = ("高", "中", "低")
PROMPT_VERSION = "three-agents-overall-three-level-v3-chinese"

COMMON_BACKGROUND = """你是上市公司财务舞弊风险识别系统中的一名智能体。
任务：评估一家企业在指定财务年度的【总体财务舞弊风险】，输出单一总体风险等级。
语言要求：所有面向人的文字内容必须使用简体中文，包括风险点、指标名称、证据、理由、摘要、关注事项和核查建议；不得输出英文句子或中英文混杂表述。JSON字段名、固定机器枚举值、规则编号、公司或产品原名，以及确有必要的标准缩写除外。财务指标优先使用中文全称，不要直接输出底稿中的英文字段名。
底稿的19条规则按照财务异常来源标记为R1（收入相关）、R2（存货及资产相关）、R3（资金与现金流相关）；这些标记用于追溯证据和组织分析。

全系统采用且仅采用以下三个有序风险等级：高 > 中 > 低。
  高：至少两项相对独立的强证据相互印证，或至少三项独立证据形成清晰完整的风险证据链，且核心异常无法被充分解释；属于高核查优先级，而非断言舞弊已经成立。
  中：存在可信异常或部分相互印证的证据，但关键环节仍待核验；也可用于已有疑点但信息不足以可靠上调或下调时。
  低：有效数据覆盖较充分，异常少或强度较弱，且已有具体反向证据或合理解释；主要财务关系与行业对比未形成重要疑点，但不能断言完全没有风险。
  缺失值、计算异常、规则不可用只能增加不确定性，不能单凭缺失就评为“低”；规则未命中不代表舞弊不存在。

铁律：
  1. 全部结论必须引用输入工作底稿内可核验的事实，不得虚构财务指标、规则、外部事实或企业背景。
  2. 无效/停用/未实现/计算异常/数据不足的规则不得作为肯定或否定舞弊风险的有效证据。
  3. 注意规则和原始指标的相关性：同一财务事实衍生出的多个规则不得简单视为独立证据；不假定存在底稿中未提供的规则权重。
  4. 风险等级反映进一步核查的优先程度，不是法律意义上的舞弊认定，也不等于舞弊概率。
  5. 如输入存在 ML 参考结果及 shap_values，须将其视为经过独立训练与验证、整体性能较好的模型所提供的重要参考信号，而不是可忽略的普通线索。必须主动核查 ML 概率、预测结论及主要 SHAP 贡献，并结合底稿判断其是否得到财务证据支持。SHAP 值表示模型对该样本预测的相对贡献方向和大小，不是因果证明；ML 结果也不是真实标签，不能脱离底稿单独认定舞弊。若你的判断与 ML 结果明显不一致，必须给出具体的底稿证据、数据局限或 SHAP 解释作为理由。没有提供 ML 结果时禁止编造。
  6. 必须严格输出单个JSON对象，字段名和枚举值必须与要求一致，不附加Markdown或说明。
"""

PROSECUTION_SYSTEM = COMMON_BACKGROUND + """
== 角色：指控智能体（prosecution）==
以风险发现和减少漏检为导向，集中提出针对【总体财务舞弊风险】的主要疑点；你只提出初步意见，不作最终裁决。
固定证据原则：
  - main_accusations 最多3条，按证据重要性排列；没有足够证据时可以少于3条或为空，不得为凑数造疑点。
  - 每条主要指控必须指出具体底稿证据、规则编号或原始指标，说明异常与潜在舞弊风险之间的联系。
  - 可以提出19条规则以外的异常线索，但必须有底稿内数据支持；尚无充分依据的仅列入 attention_items。
  - 单一命中规则、即使severity较高，初步等级最高为“中”；跨维度命中不能自动升级，需验证证据独立性。
  - 至少两项独立强证据相互印证，或至少三项独立证据形成清晰完整的风险证据链，才考虑“高”。
  - 数据不足不能被写成风险不存在，也不能无证据地推断舞弊已经发生。
  - 输入含 ML 结果时，必须把该高性能模型的预测及主要 SHAP 贡献作为重要风险线索逐项审查；ML 指向高风险时，优先检查正向 SHAP 特征能否与规则、指标和底稿事实形成证据链。不得无理由忽略或弱化 ML 信号。
  - 检查营收与回款、利润与经营现金流、存货周转、资产变动、关联交易、资金占用等证据之间的一致性，但不得分开输出舞弊类别。

严格返回下列JSON结构：
{
  "agent_role": "prosecution",
  "risk_level": "高/中/低",
  "confidence": "低/中/高",
  "main_accusations": [
    {"accusation_id": "P1", "risk_point": "", "related_rules": [], "related_indicators": [], "evidence": "", "fraud_meaning": "", "strength": "弱/中/强"}
  ],
  "attention_items": [],
  "data_limitations": []
}
"""

DEFENSE_SYSTEM = COMMON_BACKGROUND + """
== 角色：辩护智能体（defense）==
你负责质疑指控的证据可靠性、寻找合理商业解释和反向证据，目的是降低误报，而非一味否认风险。
对每一条main_accusations依据accusation_id逐条回应，围绕总体财务舞弊风险进行分析。
辩护强度：无有效辩护 / 弱辩护 / 中等辩护 / 强辩护。
风险调整建议：维持原风险 / 轻微下调 / 下调一个等级 / 驳回该指控。
固定辩护原则：
  - 只有行业常识或未经核验的推测，最高只能评“弱辩护”；没有底稿反向证据不得评“强辩护”或建议下调风险。
  - 多项独立强证据相互印证时不能简单否定；要区分同一事实重复计数与真正独立的交叉证据。
  - 标准无保留审计意见只能作为辅助，不足以单独消除财务异常。
  - 缺失、计算失败、规则不可用均不能作为“没有舞弊”的证据。
  - 允许指出有效的反向证据并建议调低风险，但最终三级等级只能由裁决智能体决定。
  - 输入含 ML 结果时，须正面回应模型预测及主要 SHAP 贡献。若认为模型信号属于误报，必须指出可核验的底稿反向证据或数据局限，不能仅以“模型可能不准确”为由否定；ML 指向低风险时也只能作为重要反向参考，不能替代底稿核查。
  - 对任何辩护必须注明证据来源；无证据则evidence_available=false。

严格返回下列JSON结构：
{
  "agent_role": "defense",
  "defense_results": [
    {
      "target_accusation_id": "",
      "original_risk_point": "",
      "defense_strength": "无有效辩护/弱辩护/中等辩护/强辩护",
      "suggested_adjustment": "维持原风险/轻微下调/下调一个等级/驳回该指控",
      "defense_reasons": [
        {"reason_type": "行业合理性/业务模式合理性/经营周期合理性/数据质量与重要性/反向证据", "reason": "", "supporting_evidence": "", "evidence_available": true}
      ],
      "remaining_concerns": [],
      "defense_summary": ""
    }
  ],
  "overall_defense": {"main_defense_points": [], "unresolved_risks": [], "data_limitations": []}
}
"""

VERDICT_SYSTEM = COMMON_BACKGROUND + """
== 角色：裁决智能体（verdict）==
你负责对指控、辩护与原始底稿进行综合审查，给出【总体财务舞弊风险】的单一三级结论。
输出内容聚焦总体风险、证据链、异议事项和核查建议。
裁决流程：
  - 逐项核对所采纳的规则与原始数据，区分独立证据和重复描述，审查数据可用性与合理反向解释。
  - 依据系统统一三级标准判定：高/中/低。等级间的差别主要是可信独立证据的强度、数量、交叉印证程度与反向证据是否充分。
  - “高”要求至少两项相对独立的强证据且未被充分解释，或至少三项独立证据形成清晰完整的风险证据链；单一规则不能直接判“高”。
  - 存在可信但未充分证实的异常时倾向“中”，并说明仍待核查事项；不可因为证据缺失直接判“低”。
  - “低”需要较充分的数据覆盖，以及异常较少、强度较弱或存在明确的一致性/反向证据，不是简单的规则零命中。
  - 将指控拆分为采纳、驳回、保留关注；辩护后残留的疑点写入minority_opinions。
  - 如果输入含 ML 参考，必须将该高性能模型的预测及 SHAP 解释纳入裁决并赋予重要参考权重，不得无理由忽略。最终等级与 ML 信号明显不一致时，verdict_summary 必须明确说明偏离原因及对应底稿证据；同时不能机械照搬模型，也不能根据 ML 分数直接推出舞弊成立。

严格返回下列JSON结构：
{
  "agent_role": "verdict",
  "final_risk_level": "高/中/低",
  "confirmed_risks": [{"risk_point": "", "basis": "", "accepted_reason": ""}],
  "rejected_risks": [{"risk_point": "", "rejected_reason": ""}],
  "watchlist_items": [{"reason": "", "required_additional_data": ""}],
  "evidence_chain": [{"evidence_summary": "", "related_rules": [], "related_indicators": [], "strength": "弱/中/强"}],
  "minority_opinions": [],
  "audit_suggestions": [{"suggestion": "", "priority": "高/中/低"}],
  "verdict_summary": ""
}
"""


def _validate_verdict_level(verdict: Dict[str, Any]) -> str:
    """校验裁决结果是否使用规定的三级风险枚举。"""
    level = verdict.get("final_risk_level")
    if not isinstance(level, str) or level.strip() not in RISK_LEVELS:
        raise ValueError(f"裁决必须返回三级总体风险之一 {RISK_LEVELS}，实际为: {level!r}")
    return level.strip()


# ════════════════════════════════════════════════════════════════
#  四、三智能体实现
# ════════════════════════════════════════════════════════════════
class ProsecutionAgent:
    def __init__(self, llm: DeepSeekClient):
        self.llm = llm

    def run(self, workpaper: Dict[str, Any]) -> Dict[str, Any]:
        evidence = slice_for_prosecution(workpaper)
        user = ("以下是结构化工作底稿（指控智能体视角）。请基于它发现总体财务舞弊风险，"
                "严格按系统要求的 JSON 输出。\n\n"
                + json.dumps(evidence, ensure_ascii=False, indent=2, default=str))
        return self.llm.chat_json(PROSECUTION_SYSTEM, user)


class DefenseAgent:
    def __init__(self, llm: DeepSeekClient):
        self.llm = llm

    def run(self, workpaper: Dict[str, Any], prosecution_result: Dict[str, Any]) -> Dict[str, Any]:
        evidence = slice_for_defense(workpaper)
        user = (
            "以下是结构化工作底稿（辩护智能体视角）与指控智能体结果。\n"
            "请逐条对应指控中的 main_accusations（按 accusation_id），判断每个风险点是否有合理解释，"
            "区分可解释与不可解释异常，并给出辩护强度与风险调整建议。严格按系统要求的 JSON 输出。\n\n"
            "【工作底稿（辩护视角）】\n"
            + json.dumps(evidence, ensure_ascii=False, indent=2, default=str)
            + "\n\n【指控智能体结果】\n"
            + json.dumps(prosecution_result, ensure_ascii=False, indent=2, default=str)
        )
        return self.llm.chat_json(DEFENSE_SYSTEM, user)


class VerdictAgent:
    def __init__(self, llm: DeepSeekClient):
        self.llm = llm

    def run(self, workpaper: Dict[str, Any],
            prosecution_result: Dict[str, Any],
            defense_result: Dict[str, Any]) -> Dict[str, Any]:
        evidence = slice_for_verdict(workpaper)
        user = (
            "以下是结构化工作底稿（裁决智能体视角）、指控结果与辩护结果。\n"
            "请判断指控是否成立、辩护是否充分，给出单一三级总体风险、证据链与核查建议。"
            "严格按系统要求的 JSON 输出。\n\n"
            "【工作底稿（裁决视角）】\n"
            + json.dumps(evidence, ensure_ascii=False, indent=2, default=str)
            + "\n\n【指控智能体结果】\n"
            + json.dumps(prosecution_result, ensure_ascii=False, indent=2, default=str)
            + "\n\n【辩护智能体结果】\n"
            + json.dumps(defense_result, ensure_ascii=False, indent=2, default=str)
        )
        return self.llm.chat_json(VERDICT_SYSTEM, user)


# ════════════════════════════════════════════════════════════════
#  五、协同编排器：负责整体流程控制，依次调用三智能体，并整合输出结果
# ════════════════════════════════════════════════════════════════
class ThreeAgentOrchestrator:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 temperature: float = DEFAULT_TEMPERATURE):
        self.llm = DeepSeekClient(api_key=api_key, model=model, temperature=temperature)
        self.prosecution = ProsecutionAgent(self.llm)
        self.defense = DefenseAgent(self.llm)
        self.verdict = VerdictAgent(self.llm)

    def analyze(self, workpaper: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        if not workpaper.get("success", True):
            return {"success": False, "error": workpaper.get("error", "底稿无效")}

        info = _public_basic_info(workpaper)
        tag = f"{info.get('company_name', '')} {info.get('stock_code', '')} {info.get('year', '')}"

        # 第二步：指控
        if verbose:
            print(f"\n[1/3] 指控智能体分析中…  {tag}")
        pros = self.prosecution.run(workpaper)

        # 第三步：辩护
        if verbose:
            print("[2/3] 辩护智能体分析中…")
        defe = self.defense.run(workpaper, pros)

        # 第四步：裁决
        if verbose:
            print("[3/3] 裁决智能体分析中…")
        verd = self.verdict.run(workpaper, pros, defe)
        _validate_verdict_level(verd)

        return {
            "success": True,
            "basic_info": info,
            "ml_reference": {
                "ml_score": (workpaper.get("risk_summary", {}) or {}).get("ml_score"),
                "overall_risk_level_ml": (workpaper.get("risk_summary", {}) or {}).get("overall_risk_level_ml"),
            },
            "prosecution": pros,
            "defense": defe,
            "verdict": verd,
        }


# ════════════════════════════════════════════════════════════════
#  六、底稿加载（文件 or 实时调用底稿生成器）
# ════════════════════════════════════════════════════════════════
def load_workpaper_from_file(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def generate_workpaper(stock_code: str, year: int) -> Dict[str, Any]:
    """调用同目录下的 workpaper_generator 生成底稿。"""
    try:
        import workpaper_generator as wg
    except ImportError:
        sys.exit("[ERROR] 未找到 workpaper_generator.py，请将本脚本与底稿生成器、数据、模型放在同目录，"
                 "或改用 --workpaper 指定已生成的底稿 JSON。")
    gen = wg.WorkpaperGenerator()
    return gen.generate(stock_code, year)


def load_frozen_ml_reference(stock_code: str, year: int) -> Dict[str, Any]:
    """用已冻结的最佳模型生成单样本预测和 SHAP；不在此处训练模型。"""
    try:
        import numpy as np
        import pandas as pd
        import shap
    except ImportError as exc:
        raise RuntimeError(f"加载单样本 ML/SHAP 参考所需依赖失败: {exc}") from exc

    root = Path(__file__).resolve().parent
    model_dir = root / "outputs" / "ml_model"
    rule_dir = root / "outputs" / "rule_model"
    model_path = model_dir / "best_model.pkl"
    metadata_path = model_dir / "model_metadata.json"
    features_path = rule_dir / "ml_features.csv"
    mapping_path = rule_dir / "sample_id_map_internal.csv"
    for path in (model_path, metadata_path, features_path, mapping_path):
        if not path.exists():
            raise FileNotFoundError(f"缺少单样本 ML 参考文件: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = list(metadata.get("feature_columns") or [])
    if not feature_columns:
        raise ValueError("model_metadata.json 缺少 feature_columns")

    mapping = pd.read_csv(mapping_path, dtype={"stock_code": "string"}, low_memory=False)
    mapping["stock_code"] = mapping["stock_code"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    mapping["financial_year"] = pd.to_numeric(mapping["financial_year"], errors="coerce")
    target = mapping.loc[
        mapping["stock_code"].eq(str(stock_code).zfill(6))
        & mapping["financial_year"].eq(int(year))
    ]
    if len(target) != 1:
        raise ValueError(f"无法唯一定位 ML 样本: stock_code={stock_code}, year={year}, matches={len(target)}")
    sample_id = str(target.iloc[0]["sample_id"])

    features = pd.read_csv(features_path, usecols=["sample_id", *feature_columns], low_memory=False)
    row = features.loc[features["sample_id"].astype(str).eq(sample_id)]
    if len(row) != 1:
        raise ValueError(f"ml_features.csv 无法唯一定位 sample_id={sample_id}")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    probability = float(model.predict_proba(row[feature_columns])[:, 1][0])
    threshold = float(metadata["classification_threshold"])
    prediction = int(probability >= threshold)
    imputer = model.named_steps["imputer"]
    estimator = model.named_steps["model"]
    transformed = imputer.transform(row[feature_columns])
    shap_values = shap.TreeExplainer(estimator).shap_values(transformed)
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.shape != (1, len(feature_columns)):
        raise ValueError(f"SHAP 返回结果形状异常: {values.shape}")
    top_indices = np.argsort(-np.abs(values[0]))[:10]
    top_shap = [
        {
            "rank": rank,
            "feature": feature_columns[index],
            "feature_value": float(transformed[0, index]),
            "shap_value": float(values[0, index]),
        }
        for rank, index in enumerate(top_indices, 1)
    ]
    return {
        "sample_id": sample_id,
        "model": metadata.get("model"),
        "ml_probability": probability,
        "ml_prediction": prediction,
        "classification_threshold": threshold,
        "shap_values": top_shap,
        "reference_note": "已冻结模型的预测和SHAP解释，仅作重要参考，不是真实标签。",
    }


# ════════════════════════════════════════════════════════════════
#  七、命令行入口
# ════════════════════════════════════════════════════════════════
def _sep(char: str = "─", width: int = 64) -> str:
    return char * width


def _print_accusations(accusations: List[Dict[str, Any]], label: str) -> None:
    """打印总体财务舞弊主要指控点列表。"""
    if not accusations:
        print(f"    {label}: 未发现主要指控点")
        return
    for acc in accusations:
        aid   = acc.get("accusation_id", "?")
        rp    = acc.get("risk_point", "")
        rules = ", ".join(acc.get("related_rules", [])) or "—"
        indic = ", ".join(acc.get("related_indicators", [])) or "—"
        evid  = acc.get("evidence", "")
        mean  = acc.get("fraud_meaning", "")
        stre  = acc.get("strength", "")
        print(f"    [{aid}] {rp}  (强度: {stre})")
        print(f"          相关规则: {rules}")
        print(f"          相关指标: {indic}")
        if evid:
            print(f"          证据: {evid}")
        if mean:
            print(f"          舞弊含义: {mean}")


def _print_defense_results(defense_results: List[Dict[str, Any]]) -> None:
    """打印辩护逐条结果。"""
    if not defense_results:
        print("    （无辩护条目）")
        return
    for dr in defense_results:
        tid    = dr.get("target_accusation_id", "?")
        rp     = dr.get("original_risk_point", "")
        dstr   = dr.get("defense_strength", "")
        adj    = dr.get("suggested_adjustment", "")
        summ   = dr.get("defense_summary", "")
        remain = dr.get("remaining_concerns", [])
        print(f"    [{tid}] {rp}")
        print(f"          辩护强度: {dstr}  →  建议: {adj}")
        # 辩护理由
        for r in dr.get("defense_reasons", []):
            rtype_r = r.get("reason_type", "")
            reason  = r.get("reason", "")
            evail   = "✓有证据" if r.get("evidence_available") else "✗无底稿证据"
            supev   = r.get("supporting_evidence", "")
            print(f"          [{rtype_r} {evail}] {reason}")
            if supev:
                print(f"            └ 证据: {supev}")
        if remain:
            print(f"          仍存疑点: {'; '.join(str(x) for x in remain)}")
        if summ:
            print(f"          辩护总结: {summ}")


def print_summary(result: Dict[str, Any]) -> None:
    """展示三智能体各自的完整结论，最后汇总裁决。"""
    if not result.get("success"):
        print(f"\n[失败] {result.get('error')}")
        return

    info = result.get("basic_info", {}) or {}
    ml   = result.get("ml_reference", {}) or {}
    pros = result.get("prosecution", {}) or {}
    defe = result.get("defense", {}) or {}
    verd = result.get("verdict", {}) or {}

    header = (f"{info.get('company_name','')}  "
              f"{info.get('stock_code','')}  {info.get('year','')}")

    # ── 标题栏 ────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print(f"  三智能体协同分析报告  |  {header}")
    print(f"  ML参考: 得分 {ml.get('ml_score','N/A')} / 等级 {ml.get('overall_risk_level_ml','N/A')}")
    print("═" * 64)

    # ════════════════════════════════════════════════════════════
    #  ① 指控智能体结论
    # ════════════════════════════════════════════════════════════
    print("\n【一】指控智能体结论")
    print(_sep())

    print(f"  ▶ 总体舞弊风险初判  等级: {pros.get('risk_level','—')}  置信度: {pros.get('confidence','—')}")
    _print_accusations(pros.get("main_accusations", []), "主要指控")

    # 关注事项
    attention = pros.get("attention_items", []) or []
    if attention:
        print(f"\n  ▶ 关注事项 ({len(attention)} 条)")
        for i, item in enumerate(attention, 1):
            print(f"    {i}. {item}")

    # 数据局限
    pros_limits = pros.get("data_limitations", []) or []
    if pros_limits:
        print(f"\n  ▶ 数据局限: {'; '.join(str(x) for x in pros_limits)}")

    # ════════════════════════════════════════════════════════════
    #  ② 辩护智能体结论
    # ════════════════════════════════════════════════════════════
    print("\n\n【二】辩护智能体结论")
    print(_sep())

    _print_defense_results(defe.get("defense_results", []))

    # 整体辩护摘要
    od = defe.get("overall_defense", {}) or {}
    main_pts = od.get("main_defense_points", []) or []
    unresolved = od.get("unresolved_risks", []) or []
    defe_limits = od.get("data_limitations", []) or []

    if main_pts:
        print(f"\n  ▶ 主要辩护论点:")
        for pt in main_pts:
            print(f"    • {pt}")
    if unresolved:
        print(f"\n  ▶ 未能化解的风险:")
        for ur in unresolved:
            print(f"    ✗ {ur}")
    if defe_limits:
        print(f"\n  ▶ 数据局限: {'; '.join(str(x) for x in defe_limits)}")

    # ════════════════════════════════════════════════════════════
    #  ③ 裁决智能体结论
    # ════════════════════════════════════════════════════════════
    print("\n\n【三】裁决智能体结论")
    print(_sep())

    print(f"  ★ 最终总体风险等级   : {verd.get('final_risk_level','—')}")

    # 采纳风险
    confirmed = verd.get("confirmed_risks", []) or []
    print(f"\n  ▶ 采纳风险点 ({len(confirmed)} 个)")
    for cr in confirmed:
        print(f"    [采纳] {cr.get('risk_point','')}")
        print(f"          依据: {cr.get('basis','')}")
        print(f"          采纳原因: {cr.get('accepted_reason','')}")

    # 驳回风险
    rejected = verd.get("rejected_risks", []) or []
    print(f"\n  ▶ 驳回风险点 ({len(rejected)} 个)")
    for rr in rejected:
        print(f"    [驳回] {rr.get('risk_point','')}")
        print(f"          驳回原因: {rr.get('rejected_reason','')}")

    # 保留关注
    watchlist = verd.get("watchlist_items", []) or []
    print(f"\n  ▶ 保留关注 ({len(watchlist)} 项)")
    for wi in watchlist:
        print(f"    [关注] {wi.get('reason','')}")
        if wi.get("required_additional_data"):
            print(f"          需补充: {wi.get('required_additional_data')}")

    # 核心证据链
    ev_chain = verd.get("evidence_chain", []) or []
    if ev_chain:
        print(f"\n  ▶ 核心证据链")
        for ev in ev_chain:
            rules = ", ".join(ev.get("related_rules", [])) or "—"
            indic = ", ".join(ev.get("related_indicators", [])) or "—"
            print(f"    [强度:{ev.get('strength','')}]")
            print(f"      {ev.get('evidence_summary','')}")
            print(f"      规则: {rules}  指标: {indic}")

    # 少数意见
    minority = verd.get("minority_opinions", []) or []
    if minority:
        print(f"\n  ▶ 少数意见")
        for mo in minority:
            print(f"    △ {mo}")

    # 核查建议
    suggestions = verd.get("audit_suggestions", []) or []
    if suggestions:
        print(f"\n  ▶ 核查建议 ({len(suggestions)} 条)")
        for sg in suggestions:
            pri = sg.get("priority", "")
            pri_icon = {"高": "🔴", "中": "🟡", "低": "🟢"}.get(pri, "•")
            print(f"    {pri_icon}[{pri}] {sg.get('suggestion','')}")

    # 裁决总结
    print(f"\n  ▶ 裁决总结:")
    print(f"    {verd.get('verdict_summary','')}")

    # ── 尾部汇总一览 ─────────────────────────────────────────────
    print("\n" + "═" * 64)
    print(f"  汇总一览  |  {header}")
    print(_sep())
    print(f"  指控  → 总体风险初判: {pros.get('risk_level','—')}")

    # 辩护整体强度统计
    strengths = [dr.get("defense_strength", "") for dr in (defe.get("defense_results") or [])]
    strong_cnt = strengths.count("强辩护")
    mid_cnt    = strengths.count("中等辩护")
    weak_cnt   = strengths.count("弱辩护")
    no_cnt     = strengths.count("无有效辩护")
    print(f"  辩护  → 强:{strong_cnt} 中:{mid_cnt} 弱:{weak_cnt} 无:{no_cnt}  "
          f"未化解风险: {len(unresolved)} 项")

    print(f"  裁决  → 总体: {verd.get('final_risk_level','—'):<8}  "
          f"采纳:{len(confirmed)} 驳回:{len(rejected)} 保留:{len(watchlist)}")
    print("═" * 64)


# ── 批量 RulePacket 接口 ──
def _legacy_workpaper(rule_packet: Dict[str, Any]) -> Dict[str, Any]:
    """把统一 RulePacket 视图补齐为本文件三智能体使用的完整底稿。"""
    rules = [x for x in rule_packet.get("rules", []) if isinstance(x, dict)]
    hits = [x for x in rules if x.get("rule_hit") == 1]
    unavailable = [
        x for x in rules
        if x.get("rule_status") in {"data_insufficient", "calculation_error"}
    ]
    return {
        "success": True,
        "rule_packet": rule_packet,
        "basic_info": {
            "sample_id": rule_packet.get("sample_id"),
            "year": rule_packet.get("financial_year"),
            "industry_code": rule_packet.get("industry_code"),
            "industry_parent": rule_packet.get("industry_parent"),
        },
        "risk_summary": {
            "hit_count": len(hits),
            "risk_flags": [x.get("rule_id") for x in hits],
            "ml_score": None,
            "overall_risk_level_ml": None,
        },
        "core_financial_metrics": {"rule_count": len(rules), "hit_count": len(hits), "rules": rules},
        "hit_rules_detail": hits,
        "evidence_chain": [
            {"rule_id": x.get("rule_id"), "evidence_id": x.get("evidence_id"),
             "evidence": x.get("evidence_text"), "severity": x.get("severity")}
            for x in hits
        ],
        "key_risk_signals": [x.get("rule_id") for x in hits],
        "peer_comparison": {"industry_code": rule_packet.get("industry_code"),
                            "industry_parent": rule_packet.get("industry_parent")},
        "non_financial_indicators": {},
        "questions_for_verification": [
            {"rule_id": x.get("rule_id"), "question": "核验该规则对应的原始凭证与业务解释。"}
            for x in hits
        ],
        "engine_status": {
            "total_rules": len(rules),
            "normal_calculated": len(rules) - len(unavailable),
            "details": {
                str(x.get("rule_id")): {"status": x.get("rule_status"), "name": x.get("rule_name", "")}
                for x in rules
            },
        },
    }


class ThreeAgentReview:
    """批量评测接口，直接复用本文件的 ThreeAgentOrchestrator。"""

    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    def analyze(self, rule_packet: Dict[str, Any]) -> Dict[str, Any]:
        packet = dict(rule_packet)
        workpaper = _legacy_workpaper(packet)
        ml_reference = packet.get("ml_reference")
        if isinstance(ml_reference, dict):
            workpaper["risk_summary"].update({
                "ml_score": ml_reference.get("ml_probability", ml_reference.get("ml_score")),
                "overall_risk_level_ml": ml_reference.get("overall_risk_level_ml"),
            })
            workpaper["ml_reference"] = ml_reference

        # 复用批量线程创建的客户端，避免再次初始化 OpenAI 客户端。
        orchestrator = ThreeAgentOrchestrator.__new__(ThreeAgentOrchestrator)
        orchestrator.llm = self.client
        orchestrator.prosecution = ProsecutionAgent(self.client)
        orchestrator.defense = DefenseAgent(self.client)
        orchestrator.verdict = VerdictAgent(self.client)
        raw = orchestrator.analyze(workpaper, verbose=False)
        verdict = raw.get("verdict") or {}
        level = _validate_verdict_level(verdict)
        return {
            "prompt_version": PROMPT_VERSION,
            "sample_id": packet.get("sample_id"),
            "rule_packet_hash": packet.get("rule_packet_hash"),
            "risk_level": level,
            "rule_packet": packet,
            "prosecution": raw.get("prosecution", {}),
            "defense": raw.get("defense", {}),
            "verdict": verdict,
            "evaluation": {
                "rule_packet_rule_count": len(packet.get("rules", [])),
                "verdict_has_evidence_chain": bool(verdict.get("evidence_chain")),
                "final_conclusion": level,
            },
            "run_metadata": {
                "run_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "model": self.client.model,
                "temperature": getattr(self.client, "temperature", DEFAULT_TEMPERATURE),
            },
        }


def main():
    parser = argparse.ArgumentParser(description="基于结构化工作底稿的三智能体协同分析")
    parser.add_argument("--workpaper", type=str, help="已生成的工作底稿 JSON 文件路径")
    parser.add_argument("--stock_code", type=str, help="股票代码（与 --year 一起实时生成底稿）")
    parser.add_argument("--year", type=int, help="年份")
    parser.add_argument("--with_ml_reference", action="store_true",
                        help="注入已冻结最佳模型的预测概率、预测结果和前10个SHAP贡献")
    parser.add_argument("--ml_prediction_override", type=int, choices=(0, 1),
                        help="仅用于敏感性实验：人工覆盖提供给智能体的ML二分类结果，并保留原始预测")
    parser.add_argument("--ml_probability_override", type=float,
                        help="仅用于敏感性实验：人工覆盖提供给智能体的ML概率（0至1），并保留原始概率")
    parser.add_argument("--out", type=str, default=None, help="结果输出 JSON 路径")
    parser.add_argument("--api_key", type=str, default=None, help="DeepSeek API Key（或用环境变量 DEEPSEEK_API_KEY）")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="模型名（deepseek-chat / deepseek-reasoner）")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("[ERROR] 未提供 API Key。请用 --api_key 或设置环境变量 DEEPSEEK_API_KEY。")

    # 加载底稿
    if args.workpaper:
        workpaper = load_workpaper_from_file(args.workpaper)
    elif args.stock_code and args.year:
        workpaper = generate_workpaper(args.stock_code, args.year)
    else:
        sys.exit("[ERROR] 请提供 --workpaper 文件，或同时提供 --stock_code 与 --year。")

    if args.with_ml_reference:
        if not (args.stock_code and args.year):
            sys.exit("[ERROR] --with_ml_reference 需要同时提供 --stock_code 与 --year。")
        ml_reference = load_frozen_ml_reference(args.stock_code, args.year)
        if args.ml_probability_override is not None:
            if not 0.0 <= args.ml_probability_override <= 1.0:
                sys.exit("[ERROR] --ml_probability_override 必须位于0至1之间。")
            ml_reference["original_ml_probability"] = float(ml_reference["ml_probability"])
            ml_reference["ml_probability"] = float(args.ml_probability_override)
            ml_reference["manual_probability_override"] = True
        if args.ml_prediction_override is not None:
            original_prediction = int(ml_reference["ml_prediction"])
            ml_reference["original_ml_prediction"] = original_prediction
            ml_reference["ml_prediction"] = int(args.ml_prediction_override)
            ml_reference["manual_prediction_override"] = True
            ml_reference["override_note"] = (
                "敏感性实验人工覆盖：当前ml_prediction不是模型按概率和阈值生成的原始结果；"
                "original_ml_prediction保留真实模型结果。"
            )
        if ml_reference.get("manual_prediction_override") or ml_reference.get("manual_probability_override"):
            ml_reference["reference_note"] = (
                "人工覆盖的ML概率/分类敏感性情景；original_ml_probability和original_ml_prediction"
                "保留真实模型输出，SHAP仍来自原始模型，不是真实标签。"
            )
            ml_reference["override_note"] = (
                "敏感性实验人工覆盖：当前ml_probability或ml_prediction可能不是模型原始输出；"
                "请使用original_ml_probability和original_ml_prediction追溯真实结果。"
            )
        workpaper["ml_reference"] = ml_reference
        risk_summary = workpaper.setdefault("risk_summary", {})
        risk_summary["ml_score"] = ml_reference["ml_probability"]
        risk_summary["overall_risk_level_ml"] = (
            "预警" if ml_reference["ml_prediction"] == 1 else "不预警"
        )
        print(
            f"[ML] {ml_reference['model']} probability={ml_reference['ml_probability']:.6f}, "
            f"threshold={ml_reference['classification_threshold']:.6f}, "
            f"prediction={ml_reference['ml_prediction']}, SHAP特征={len(ml_reference['shap_values'])}"
        )
        if ml_reference.get("manual_prediction_override"):
            print(
                f"[ML][OVERRIDE] 原始预测={ml_reference['original_ml_prediction']}，"
                f"情景预测={ml_reference['ml_prediction']}；仅用于敏感性实验"
            )
        if ml_reference.get("manual_probability_override"):
            print(
                f"[ML][OVERRIDE] 原始概率={ml_reference['original_ml_probability']:.6f}，"
                f"情景概率={ml_reference['ml_probability']:.6f}；SHAP仍为原始模型解释"
            )
    elif args.ml_prediction_override is not None or args.ml_probability_override is not None:
        sys.exit("[ERROR] ML覆盖参数必须与 --with_ml_reference 同时使用。")

    orchestrator = ThreeAgentOrchestrator(
        api_key=api_key, model=args.model, temperature=args.temperature,
    )
    result = orchestrator.analyze(workpaper)

    if args.out:
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\n[完成] 结果已保存至: {args.out}")

    print_summary(result)


if __name__ == "__main__":
    main()
