#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""财务舞弊识别工作底稿生成器。

数据与规则边界：
  1. 原始数据只读取 ``汇总数据.csv``。
  2. 19 条规则、行业年度阈值和证据字段由 ``rule_model.py`` 提供。

底稿模块：
  A. sample_basic_info         样本基本信息
  B. financial_metrics         核心财务指标
  C. industry_benchmarks       行业基准
  D. rule_hit_results          规则命中结果（19 条）
  E. governance_non_financial  治理与非财务信息
  F. data_quality              数据质量说明
  G. agent_permissions         智能体读权限说明
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import rule_model as rule_engine


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "汇总数据.csv"
RULE_OUTPUT_DIR = ROOT / "outputs" / "rule_model"
THRESHOLD_TABLE_PATH = RULE_OUTPUT_DIR / "threshold_table.csv"
RULE_CODES = [code for code, *_ in rule_engine.RULE_CATALOG]


def safe_div(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    try:
        if pd.isna(a) or pd.isna(b) or float(b) == 0:
            return None
        return float(a) / float(b)
    except (TypeError, ValueError):
        return None


def nan_to_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def fmt_billion(value: Any) -> float | None:
    value = nan_to_none(value)
    if value is None:
        return None
    return round(float(value) / 1e8, 2)


def scale_group(total_assets: Any) -> str:
    total_assets = nan_to_none(total_assets)
    if total_assets is None:
        return "未知"
    if total_assets >= 1e10:
        return "大型（资产>=100亿）"
    if total_assets >= 1e9:
        return "中大型（10亿<=资产<100亿）"
    if total_assets >= 1e8:
        return "中型（1亿<=资产<10亿）"
    return "小型（资产<1亿）"


class WorkpaperGenerator:
    """从汇总数据与统一 RulePacket 构建工作底稿。"""

    def __init__(self) -> None:
        self.df_prepared: pd.DataFrame | None = None
        self.threshold_table: pd.DataFrame | None = None
        self.rule_lookup: rule_engine.StatLookup | None = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}")

        df, metadata = rule_engine.load_data(DATA_PATH)
        self.df_prepared = rule_engine._derived_threshold_metrics(df)
        if "cash_received_growth_yoy" in self.df_prepared.columns:
            self.df_prepared["cash_received_growth_m"] = (
                rule_engine._normalize_growth(
                    self.df_prepared["cash_received_growth_yoy"]
                )
            )
        print(
            f"[INFO] 汇总数据加载成功，共 {metadata['rows']} 条记录，"
            f"年份 {metadata['year_min']}-{metadata['year_max']}"
        )

        if THRESHOLD_TABLE_PATH.exists():
            self.threshold_table = pd.read_csv(THRESHOLD_TABLE_PATH, low_memory=False)
            print(f"[INFO] 行业年度阈值加载成功: {THRESHOLD_TABLE_PATH}")
        else:
            print("[WARN] 未找到规则阈值输出，正在按 rule_model.py 的统一流程构建")
            lockbox_spec, lockbox_firms = rule_engine._load_lockbox_spec()
            split = rule_engine._company_split(
                self.df_prepared, lockbox_spec, lockbox_firms
            )
            self.threshold_table = rule_engine.build_threshold_table(
                self.df_prepared, split.eq("train")
            )

        self.rule_lookup = rule_engine.StatLookup(self.threshold_table)
        self._loaded = True

    def _build_rule_packet(self, row: pd.Series) -> dict[str, Any]:
        if self.rule_lookup is None:
            raise RuntimeError("规则阈值尚未加载")

        rules = [
            rule_engine._one_rule(row, self.rule_lookup, code)
            for code in RULE_CODES
        ]
        for rule in rules:
            rule["evidence_text"] = rule_engine._evidence_text(rule)

        packet = {
            "rule_packet_version": rule_engine.VERSION,
            "threshold_version": rule_engine.THRESHOLD_VERSION,
            "sample_id": row["sample_id"],
            "financial_year": int(row["financial_year"]),
            "industry_code": str(row["industry_code"]),
            "industry_parent": str(row["industry_parent"]),
            "rules": rules,
        }
        canonical = json.dumps(
            packet, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        packet["rule_packet_hash"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        return packet

    def _industry_stat(
        self, row: pd.Series, metric: str, statistic: str
    ) -> Any:
        if self.rule_lookup is None:
            return None
        stat = self.rule_lookup.get(
            int(row["financial_year"]),
            str(row["industry_code"]),
            str(row["industry_parent"]),
            metric,
        )
        return nan_to_none(stat.get(statistic)) if stat else None

    def _benchmark_metadata(self, row: pd.Series, metric: str) -> dict[str, Any]:
        """返回与 RulePacket 同源的行业层级、样本量和阈值来源。"""
        if self.threshold_table is None:
            return {"threshold_source": None, "industry_level": None, "n_obs": None}
        year = int(row["financial_year"])
        industry_code = str(row["industry_code"])
        industry_parent = str(row["industry_parent"])
        table = self.threshold_table
        candidates = table[
            table["financial_year"].eq(year)
            & table["metric_name"].eq(metric)
            & (
                table["industry_code"].astype("string").eq(industry_code)
                | table["industry_parent"].astype("string").eq(industry_parent)
            )
        ]
        if candidates.empty:
            return {"threshold_source": None, "industry_level": None, "n_obs": None}
        item = candidates.iloc[0]
        return {
            "threshold_source": nan_to_none(item.get("threshold_source")),
            "industry_level": nan_to_none(item.get("industry_level")),
            "n_obs": nan_to_none(item.get("n_obs")),
            "n_min": nan_to_none(item.get("n_min")),
            "threshold_available": nan_to_none(item.get("threshold_available")),
        }

    def generate(self, stock_code: str, year: int) -> dict[str, Any]:
        self._ensure_loaded()
        if self.df_prepared is None:
            raise RuntimeError("汇总数据尚未加载")

        stock_code = str(stock_code).zfill(6)
        year = int(year)
        print(f"\n[INFO] 正在生成工作底稿: {stock_code} {year}")

        mask = (
            self.df_prepared["stock_code"].eq(stock_code)
            & self.df_prepared["financial_year"].eq(year)
        )
        if not mask.any():
            return {
                "success": False,
                "error": f"汇总数据中未找到 {stock_code} {year}",
            }

        row = self.df_prepared.loc[mask].iloc[0]
        rule_packet = self._build_rule_packet(row)
        return self._build_workpaper(row, rule_packet)

    def _build_workpaper(
        self, row: pd.Series, rule_packet: dict[str, Any]
    ) -> dict[str, Any]:
        stock_code = str(row.get("stock_code", "")).zfill(6)
        year = int(row["financial_year"])
        company_name = str(row.get("company_name", ""))
        industry_code = str(row.get("industry_code", ""))
        industry_name = str(row.get("industry_name", ""))
        total_assets = row.get("total_assets")
        rule_results = rule_packet["rules"]
        hit_rules = [rule for rule in rule_results if rule["rule_hit"] == 1]

        module_a = {
            "_module": "A. 样本基本信息",
            "_note": "本模块不向智能体披露公司名称、股票代码等可识别身份信息。",
            "sample_id": rule_packet["sample_id"],
            "year": year,
            "industry_code": industry_code,
            "industry_name": industry_name,
            "scale_group": scale_group(total_assets),
            "_internal_stock_code": stock_code,
            "_internal_company_name": company_name,
        }

        # B 模块完整保留汇总数据中的财务报表、披露指标和规则所需派生指标。
        # 所有缺失值均保留为 None，不把缺失误写成真实的 0。
        balance_sheet_fields = [
            "cash", "trading_financial_assets", "notes_receivable_net",
            "accounts_receivable_net", "prepayments_net", "inventory_net",
            "current_assets_total", "investment_property_net", "fixed_assets_net",
            "construction_in_progress_net", "intangible_assets_net", "goodwill_net",
            "non_current_assets_total", "total_assets", "short_term_borrowings",
            "notes_payable", "accounts_payable", "advance_from_customers",
            "employee_benefits_payable", "taxes_fees_payable", "interest_payable",
            "dividends_payable", "current_liabilities_total", "long_term_borrowings",
            "long_term_payables", "total_liabilities", "paid_in_capital",
            "other_equity_instruments", "capital_reserve", "surplus_reserve",
            "retained_earnings", "equity_attributable_to_parent", "total_owners_equity",
            "total_liabilities_and_equity",
        ]
        income_statement_fields = [
            "total_operating_revenue", "operating_revenue", "net_interest_income",
            "other_operating_revenue", "total_operating_cost", "operating_cost",
            "taxes_and_surcharges", "selling_expenses", "administrative_expenses",
            "research_development_expenses", "finance_expenses", "interest_expense",
            "interest_income", "other_gains", "investment_income", "operating_profit",
            "non_operating_revenue", "non_operating_expenses", "total_profit",
            "income_tax_expense", "net_profit",
        ]
        cash_flow_fields = [
            "cash_received_from_sales", "operating_cash_inflows", "cash_paid_for_purchases",
            "operating_cash_outflows", "operating_cash_flow_net", "investing_cash_flow_net",
            "financing_cash_flow_net", "cash_equivalents_increase_net",
            "cash_equivalents_beginning", "cash_equivalents_ending",
        ]
        disclosed_fields = [
            "non_recurring_gain_loss", "net_profit_deducted_nonrecurring", "weighted_avg_roe",
            "weighted_avg_roe_deducted", "basic_eps_deducted", "operating_cash_flow_per_share",
            "bps", "basic_eps", "diluted_eps",
        ]
        ratio_fields = [
            "gross_profit_margin", "selling_expense_ratio", "receivables_turnover_ratio",
            "receivables_turnover_days", "inventory_turnover_ratio", "inventory_turnover_days",
            "revenue_growth", "interest_liab_ratio",
        ]
        derived_fields = [
            "revenue_growth_m", "receivable_growth_m", "inventory_growth_m", "cost_growth_m",
            "gross_margin_m", "sales_expense_ratio_m", "sales_expense_drop_m",
            "net_profit_growth_m", "cash_collection_ratio_m", "ocf_profit_ratio_m",
            "inventory_to_assets_m", "inventory_turnover_m", "construction_growth_m",
            "fixed_assets_growth_m", "construction_growth_lag_m", "cash_to_assets_m",
            "short_loan_to_liab_m", "prepayment_to_assets_m", "prepayment_growth_m",
            "related_party_to_assets_m", "interest_income_to_cash_m", "financing_to_assets_m",
            "neg_cfo_3y_m", "receivable_minus_revenue_growth_m", "inventory_minus_cost_growth_m",
        ]
        module_b = {
            "_module": "B. 核心财务指标",
            "unit_note": "原始金额字段单位为元；scale_billion_yuan为亿元；比率按原始数据口径保留。",
            "scale_billion_yuan": {
                field: fmt_billion(row.get(field))
                for field in ("cash", "total_assets", "operating_revenue", "net_profit", "operating_cash_flow_net")
            },
            "balance_sheet": {field: nan_to_none(row.get(field)) for field in balance_sheet_fields},
            "income_statement": {field: nan_to_none(row.get(field)) for field in income_statement_fields},
            "cash_flow_statement": {field: nan_to_none(row.get(field)) for field in cash_flow_fields},
            "disclosed_financial_metrics": {field: nan_to_none(row.get(field)) for field in disclosed_fields},
            "reported_ratios": {field: nan_to_none(row.get(field)) for field in ratio_fields},
            "derived_rule_metrics": {field: nan_to_none(row.get(field)) for field in derived_fields},
        }

        benchmark_metrics = sorted({key.split(":", 1)[0] for rule in rule_results for key in rule.get("threshold_values", {}).keys()})
        module_c = {
            "_module": "C. 行业年度基准",
            "_note": "基准完全采用 rule_model.py 的行业×财务年度阈值表；每项同时保留样本量、来源和分位数。",
            "threshold_version": rule_packet["threshold_version"],
            "industry_code": industry_code,
            "industry_parent": str(row.get("industry_parent", "")),
            "financial_year": year,
            "metrics": {metric: {statistic: self._industry_stat(row, metric, statistic) for statistic in ("mean", "median", "p10", "p25", "p75", "p90")} for metric in benchmark_metrics},
            "source_metadata": {metric: self._benchmark_metadata(row, metric) for metric in benchmark_metrics},
        }

        by_category: dict[str, dict[str, Any]] = {}
        for category in dict.fromkeys(rule["category"] for rule in rule_results):
            category_rules = [
                rule for rule in rule_results if rule["category"] == category
            ]
            category_hits = [
                rule for rule in category_rules if rule["rule_hit"] == 1
            ]
            by_category[category] = {
                "rule_count": len(category_rules),
                "hit_count": len(category_hits),
                "hit_rule_ids": [rule["rule_id"] for rule in category_hits],
            }

        unavailable_rules = [
            rule
            for rule in rule_results
            if rule["rule_status"] in {"data_insufficient", "calculation_error"}
        ]
        module_d = {
            "_module": "D. 规则命中结果",
            "_note": (
                "19条规则结果完整取自rule_model.py生成的统一RulePacket；"
                "severity表示单条规则异常强弱，不表示底稿整体等级。"
            ),
            "rule_source": "rule_model.py",
            "rule_packet_version": rule_packet["rule_packet_version"],
            "rule_packet_hash": rule_packet["rule_packet_hash"],
            "total_rules": len(rule_results),
            "available_rules": len(rule_results) - len(unavailable_rules),
            "unavailable_rules": len(unavailable_rules),
            "hit_count": len(hit_rules),
            "by_category": by_category,
            "all_rules_detail": rule_results,
        }

        module_e = self._build_governance_module(row)

        status_counts = {
            status: sum(1 for rule in rule_results if rule["rule_status"] == status)
            for status in (
                "normal",
                "watch",
                "data_insufficient",
                "calculation_error",
            )
        }
        missing_fields = sorted(
            {
                field
                for rule in unavailable_rules
                for field in rule.get("missing_fields", [])
            }
        )
        module_f = {
            "_module": "F. 数据质量说明",
            "data_source": DATA_PATH.name,
            "rule_source": "rule_model.py",
            "rule_packet_version": rule_packet["rule_packet_version"],
            "threshold_version": rule_packet["threshold_version"],
            "rule_packet_hash": rule_packet["rule_packet_hash"],
            "rule_status_counts": status_counts,
            "unusable_rules": [
                {
                    "rule_id": rule["rule_id"],
                    "rule_name": rule["rule_name"],
                    "rule_status": rule["rule_status"],
                    "missing_fields": rule.get("missing_fields", []),
                }
                for rule in unavailable_rules
            ],
            "missing_fields": missing_fields,
            "governance_field_status": module_e.get("field_status", {}),
            "governance_missing_fields": module_e.get("missing_fields", []),
        }

        module_g = {
            "_module": "G. 智能体读权限说明",
            "_note": (
                "A、B、C、E作为工作底稿留存信息；三智能体主流程只读取"
                "统一规则结果和数据质量字段。"
            ),
            "prosecution_agent": {
                "readable_modules": ["D", "F"],
                "hidden_fields": [
                    "A._internal_stock_code",
                    "A._internal_company_name",
                ],
                "focus": "从规则命中、异常强弱、阈值偏离和证据编号中提出主要指控。",
            },
            "defense_agent": {
                "readable_modules": ["D", "F", "prosecution_output"],
                "hidden_fields": [
                    "A._internal_stock_code",
                    "A._internal_company_name",
                ],
                "focus": "依据同一规则结果审查临界命中、数据不足和合理解释。",
            },
            "verdict_agent": {
                "readable_modules": [
                    "D",
                    "F",
                    "prosecution_output",
                    "defense_output",
                ],
                "hidden_fields": [
                    "A._internal_stock_code",
                    "A._internal_company_name",
                ],
                "focus": "综合双方意见，引用规则证据并形成核查结论与建议。",
            },
        }

        return {
            "success": True,
            "version": "v5.0-rulepacket",
            "data_source": DATA_PATH.name,
            "rule_source": "rule_model.py",
            "A_sample_basic_info": module_a,
            "B_financial_metrics": module_b,
            "C_industry_benchmarks": module_c,
            "D_rule_hit_results": module_d,
            "E_governance_non_financial": module_e,
            "F_data_quality": module_f,
            "G_agent_permissions": module_g,
        }

    def _build_governance_module(self, row: pd.Series) -> dict[str, Any]:
        audit_non_std = nan_to_none(row.get("audit_non_standard"))
        audit_risk = nan_to_none(row.get("audit_opinion_risk"))
        pledge_risk = nan_to_none(row.get("pledge_high_risk"))
        core_tech = nan_to_none(row.get("core_tech_turnover"))
        customer_conc = nan_to_none(row.get("customer_concentration"))
        supplier_conc = nan_to_none(row.get("supplier_concentration"))
        supply_conc = nan_to_none(row.get("supply_chain_concentration"))
        related_amount = nan_to_none(row.get("related_party_transaction_amount"))

        audit_risk_int = int(audit_risk) if audit_risk is not None else None
        audit_opinion_map = {
            0: "标准无保留意见",
            1: "带强调事项段无保留意见",
            2: "保留意见",
            3: "无法表示意见",
            4: "否定意见",
        }
        audit_desc = audit_opinion_map.get(audit_risk_int) if audit_risk_int is not None else None

        signal_flags: list[str] = []
        if audit_non_std == 1 or (audit_risk_int is not None and audit_risk_int > 0):
            signal_flags.append(f"非标审计意见（{audit_desc}）")
        if pledge_risk == 1:
            signal_flags.append("控股股东高质押")
        if core_tech == 1:
            signal_flags.append("关键技术人员发生变动")
        if customer_conc is not None and customer_conc > 50:
            signal_flags.append(f"客户集中度偏高（{customer_conc:.1f}%）")
        if supplier_conc is not None and supplier_conc > 50:
            signal_flags.append(f"供应商集中度偏高（{supplier_conc:.1f}%）")

        related_ratio = safe_div(related_amount, row.get("total_assets"))
        concentration_flag = bool(
            (customer_conc is not None and customer_conc > 50)
            or (supplier_conc is not None and supplier_conc > 50)
        )

        governance_fields = [
            "core_tech_turnover", "audit_non_standard", "audit_opinion_risk",
            "pledge_high_risk", "customer_concentration", "supplier_concentration",
            "supply_chain_concentration", "related_party_transaction_amount",
            "regulatory_letter", "inquiry_letter", "major_litigation",
            "is_shareholder_pledge", "is_pledge_ratio_over30", "is_first_holder_pledge",
            "is_first_holder_pledge_over30", "is_controller_pledge",
            "is_controller_pledge_over30", "is_shareholder_frozen", "is_freeze_ratio_over30",
            "is_first_holder_frozen", "is_first_holder_freeze_over30", "is_controller_frozen",
            "is_controller_freeze_over30", "is_punish_investigate", "punish_investigate_persons",
            "punish_investigate_count", "is_guarantee_overdue", "guarantee_overdue_count",
            "is_judicial_anomaly", "judicial_anomaly_count", "regulatory_letter_count",
            "delay_inquiry_count", "is_tmt_resign", "tmt_resign_count", "is_tmt_reduce",
            "tmt_reduce_count", "is_acquisition", "acquired_complete_num", "acquired_fail_num",
            "acquiring_num", "is_excessive_goodwill", "is_goodwill_impair_abnormal",
            "is_audit_opinion_abnormal", "audit_opinion_date", "audit_opinion_type",
            "is_intcontrol_audit_abnormal", "intcontrol_audit_date", "intcontrol_audit_type",
            "is_report_anomaly", "report_attention", "analyst_attention",
            "is_bank_account_frozen", "is_price_risk", "is_market_value_risk", "is_account_error_change",
        ]
        field_status = {
            field: ("available" if field in row.index else "not_in汇总数据")
            for field in governance_fields
        }
        missing_fields = [field for field, status in field_status.items() if status != "available"]

        return {
            "_module": "E. 治理与非财务信息",
            "_note": "治理指标仅作辅助核查信息，不并入19条规则命中结果；汇总数据未覆盖的字段保留为缺失并列入待接入清单。",
            "raw_available_metrics": {
                field: nan_to_none(row.get(field))
                for field, status in field_status.items()
                if status == "available"
            },
            "field_status": field_status,
            "missing_fields": missing_fields,
            "audit_quality": {
                "is_non_standard": bool(audit_non_std == 1),
                "opinion_risk_code": audit_risk_int,
                "opinion_desc": audit_desc,
                "risk_flag": bool(audit_non_std == 1 or (audit_risk_int is not None and audit_risk_int > 0)),
                "note": (
                    "审计意见为标准无保留意见，可作为辅助反向证据。"
                    if audit_non_std == 0 and audit_risk_int == 0
                    else "审计意见字段缺失，暂不作判断。"
                    if audit_desc is None
                    else f"审计意见为{audit_desc}，需关注财务报告可信度。"
                ),
            },
            "ownership_risk": {
                "pledge_high_risk": bool(pledge_risk == 1),
                "risk_flag": bool(pledge_risk == 1),
                "note": (
                    "控股股东质押指标为缺失，暂不作判断。"
                    if pledge_risk is None
                    else "控股股东质押比例在可控范围内。"
                    if pledge_risk == 0
                    else "控股股东存在高比例质押，需关注控制权和资金压力。"
                ),
            },
            "management_stability": {
                "core_tech_turnover": bool(core_tech == 1),
                "risk_flag": bool(core_tech == 1),
                "note": (
                    "关键技术人员变动指标为缺失，暂不作判断。"
                    if core_tech is None
                    else "关键技术人员未发生重大变动。"
                    if core_tech == 0
                    else "关键技术人员发生变动，需关注团队稳定性。"
                ),
            },
            "related_party_transaction": {
                "amount_billion": fmt_billion(related_amount),
                "to_assets": related_ratio,
                "note": (
                    "关联交易规模较大，需结合R3-04规则结果核查。"
                    if related_ratio is not None and related_ratio > 0.10
                    else "关联交易规模未超过本模块的关注比例。"
                ),
            },
            "supply_chain": {
                "customer_concentration": customer_conc,
                "supplier_concentration": supplier_conc,
                "supply_chain_concentration": supply_conc,
                "risk_flag": concentration_flag,
                "note": (
                    "供应链集中度未触发本模块关注条件。"
                    if not concentration_flag
                    else "供应链集中度偏高，需关注客户或供应商依赖。"
                ),
            },
            "summary": {
                "risk_count": len(signal_flags),
                "risk_flags": signal_flags,
                "note": (
                    "治理与非财务指标未发现明显关注信号。"
                    if not signal_flags
                    else f"治理与非财务指标发现{len(signal_flags)}个关注信号："
                    + "；".join(signal_flags)
                    + "。"
                ),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成财务舞弊识别工作底稿")
    parser.add_argument("--stock_code", type=str, help="股票代码（如 000651）")
    parser.add_argument("--year", type=int, help="年份（如 2023）")
    parser.add_argument("--output", type=str, default=None, help="输出JSON文件路径")
    args = parser.parse_args()

    if args.stock_code and args.year:
        stock_code = args.stock_code
        year = args.year
    else:
        print("=" * 50)
        print("  财务舞弊识别 - 工作底稿生成器")
        print("  汇总数据 + 19条统一规则 + A-G模块")
        print("=" * 50)
        stock_code = input("请输入股票代码（如 000651）: ").strip()
        year = int(input("请输入年份（如 2023）: ").strip())

    result = WorkpaperGenerator().generate(stock_code, year)
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n工作底稿已保存至: {output_path}")
    else:
        print("\n" + "=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
