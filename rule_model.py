#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rule engine for the financial-fraud risk identification study.

This module implements the 19-rule layer described in the implementation plan:

    raw summary data -> derived indicators -> industry x financial-year
    thresholds -> 19 rules -> one canonical RulePacket per company-year

The engine deliberately does not create an ML score, an agent verdict, or a
weighted rule score.  It exports numeric features and an auditable RulePacket
that can be sent independently to the ML route and the three-agent route.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "汇总数据.csv"
LOCKBOX_PATH = ROOT / "抽样" / "锁箱样本_1比3匹配_最终结果.xlsx"
OUT_DIR = ROOT / "outputs" / "rule_model"
VERSION = "rule_engine_v1.2_20260917"
THRESHOLD_VERSION = "industry_year_train_v2_stratified_nmin30"
RULE_FORMULA_VERSION = "rule_formula_freeze_v1.1_20260917"
ANON_SAMPLE_VERSION = "sample_id_salted_sha256_v1"
# This is a pseudonymisation salt for stable sample IDs.  The real stock code
# is never written to RulePacket or model-facing feature files; the mapping is
# written only to the internal evaluation sidecar.
ANON_SAMPLE_SALT = "fraud-rule-packet-2026-v1"
N_MIN = 30
SPLIT_SEED = 20260820
TRAIN_SHARE = 0.80


RULE_CATALOG = [
    ("R1-01", "收入舞弊", "应收账款增速显著背离营收增速", "应收账款增速-营收增速 > 行业年度P75"),
    ("R1-02", "收入舞弊", "销售回款率偏低", "销售回款率 < 行业年度P25 且营收增速 > 行业年度P50"),
    ("R1-03", "收入舞弊", "毛利率显著高于行业均值", "毛利率 > 行业年度P90"),
    ("R1-04", "收入舞弊", "净利润增长但经营现金流走弱", "净利润增速 > 行业年度P75 且 OCF/净利润 < 行业年度P25"),
    ("R1-05", "收入舞弊", "营收增速显著高于行业均值", "营收增速 > 行业年度P90"),
    ("R1-06", "收入舞弊", "销售费用率逆势下降", "营收增速 > 行业年度P75 且销售费用率降幅 > 行业年度P75"),
    ("R2-01", "存货异常", "存货增速显著高于营业成本增速", "存货增速-成本增速 > 行业年度P75"),
    ("R2-02", "存货异常", "存货周转率显著低于行业水平", "存货周转率 < 行业年度P25"),
    ("R2-03", "存货异常", "存货高企且周转偏弱", "存货/资产 > 行业年度P75 且存货周转率 < 行业年度P25"),
    ("R2-04", "存货异常", "存货占总资产比例异常偏高", "存货/资产 > 行业年度P90"),
    ("R2-05", "存货异常", "在建工程长期挂账不转固", "在建工程增速连续两期 > P75 且固定资产增速 < P25"),
    ("R3-01", "资金异常", "存贷双高", "现金/资产 > 行业年度P75 且短借/负债 > 行业年度P75"),
    ("R3-02", "资金异常", "往来款/预付款异常占用", "预付款/资产 > P75 或预付款增速 > P75"),
    ("R3-03", "资金异常", "经营现金流与利润背离", "净利润为正且 OCF/净利润 < 行业年度P25"),
    ("R3-04", "资金异常", "关联交易金额占比偏高", "关联交易金额/资产 > 行业年度P90"),
    ("R3-05", "资金异常", "货币资金与利息收入不匹配", "现金/资产 > P75 且利息收入/现金 < P25"),
    ("R3-06", "资金异常", "筹资现金流持续流入但业务无明显扩张", "筹资现金流/资产 > P75、营收增速 < P25 且 OCF≤0"),
    ("R3-07", "资金异常", "经营现金流与净利润背离（行业分位数）", "净利润为正且 OCF<0 或 OCF/净利润 < 行业年度P10"),
    ("R3-08", "资金异常", "连续三年经营性现金流为负", "连续三年OCF为负；当年仍有利润或营收增长时提高异常程度"),
]
RULE_NAMES = {x[0]: x[2] for x in RULE_CATALOG}
RULE_CATEGORY = {x[0]: x[1] for x in RULE_CATALOG}

# Frozen formula crosswalk.  The supplied rule document states business intent
# but does not provide a complete machine-readable formula for every original
# rule.  These expressions therefore freeze the implementable interpretation
# used by this run, while preserving the document's direction and population.
RULE_FORMULA_FREEZE = {
    "R1-01": {"expression": "receivable_growth_m - revenue_growth_m > P75(receivable_minus_revenue_growth_m)", "inputs": "receivable_growth_m; revenue_growth_m", "thresholds": "P75", "boundary": ">", "source_alignment": "规则文档：应收增速背离营收增速"},
    "R1-02": {"expression": "cash_collection_ratio_m < P25(cash_collection_ratio_m) AND revenue_growth_m > P50(revenue_growth_m)", "inputs": "cash_received_from_sales; operating_revenue; revenue_growth_m", "thresholds": "P25; P50", "boundary": "<; >", "source_alignment": "规则文档：营收增长但销售回款率偏低"},
    "R1-03": {"expression": "gross_margin_m > P90(gross_margin_m)", "inputs": "gross_profit_margin or operating_revenue; operating_cost", "thresholds": "P90", "boundary": ">", "source_alignment": "规则文档：毛利率显著高于行业水平"},
    "R1-04": {"expression": "net_profit_growth_m > P75(net_profit_growth_m) AND ocf_profit_ratio_m < P25(ocf_profit_ratio_m)", "inputs": "net_profit; operating_cash_flow_net", "thresholds": "P75; P25", "boundary": ">; <", "source_alignment": "规则文档：净利润增长但经营现金流走弱"},
    "R1-05": {"expression": "revenue_growth_m > P90(revenue_growth_m)", "inputs": "revenue_growth_yoy or operating_revenue", "thresholds": "P90", "boundary": ">", "source_alignment": "规则文档：营收增速显著高于行业水平"},
    "R1-06": {"expression": "revenue_growth_m > P75(revenue_growth_m) AND sales_expense_drop_m > P75(sales_expense_drop_m)", "inputs": "revenue_growth_m; selling_expense_ratio", "thresholds": "P75; P75", "boundary": ">; >", "source_alignment": "规则文档：收入增长而销售费用率逆势下降"},
    "R2-01": {"expression": "inventory_growth_m - cost_growth_m > P75(inventory_minus_cost_growth_m)", "inputs": "inventory_growth_m; cost_growth_m", "thresholds": "P75", "boundary": ">", "source_alignment": "规则文档：存货增速显著高于营业成本增速"},
    "R2-02": {"expression": "inventory_turnover_m < P25(inventory_turnover_m)", "inputs": "inventory_turnover_ratio or operating_cost/inventory_net", "thresholds": "P25", "boundary": "<", "source_alignment": "规则文档：存货周转率显著低于行业水平"},
    "R2-03": {"expression": "inventory_to_assets_m > P75(inventory_to_assets_m) AND inventory_turnover_m < P25(inventory_turnover_m)", "inputs": "inventory_net; total_assets; inventory_turnover_m", "thresholds": "P75; P25", "boundary": ">; <", "source_alignment": "规则文档：存货高企且周转偏弱"},
    "R2-04": {"expression": "inventory_to_assets_m > P90(inventory_to_assets_m)", "inputs": "inventory_net; total_assets", "thresholds": "P90", "boundary": ">", "source_alignment": "规则文档：存货占总资产比例异常偏高"},
    "R2-05": {"expression": "construction_growth_m > P75 AND construction_growth_lag_m > P75 AND fixed_assets_growth_m < P25", "inputs": "construction_in_progress_net; fixed_assets_net", "thresholds": "P75; P25", "boundary": ">; >; <", "source_alignment": "规则文档：在建工程长期挂账不转固"},
    "R3-01": {"expression": "cash_to_assets_m > P75(cash_to_assets_m) AND short_loan_to_liab_m > P75(short_loan_to_liab_m)", "inputs": "cash; total_assets; short_term_borrowings; total_liabilities", "thresholds": "P75; P75", "boundary": ">; >", "source_alignment": "规则文档：存贷双高"},
    "R3-02": {"expression": "prepayment_to_assets_m > P75(prepayment_to_assets_m) OR prepayment_growth_m > P75(prepayment_growth_m)", "inputs": "prepayments_net; total_assets", "thresholds": "P75; P75", "boundary": ">; >", "source_alignment": "规则文档：往来款/预付款异常占用"},
    "R3-03": {"expression": "net_profit > 0 AND ocf_profit_ratio_m < P25(ocf_profit_ratio_m)", "inputs": "net_profit; operating_cash_flow_net", "thresholds": "P25", "boundary": ">0; <", "source_alignment": "规则文档：经营现金流与利润背离"},
    "R3-04": {"expression": "related_party_to_assets_m > P90(related_party_to_assets_m)", "inputs": "related_party_transaction_amount; total_assets", "thresholds": "P90", "boundary": ">", "source_alignment": "规则文档：关联交易金额占比偏高"},
    "R3-05": {"expression": "cash_to_assets_m > P75(cash_to_assets_m) AND interest_income_to_cash_m < P25(interest_income_to_cash_m)", "inputs": "cash; interest_income; total_assets", "thresholds": "P75; P25", "boundary": ">; <", "source_alignment": "规则文档：货币资金与利息收入不匹配"},
    "R3-06": {"expression": "financing_to_assets_m > P75 AND revenue_growth_m < P25 AND operating_cash_flow_net <= 0", "inputs": "financing_cash_flow_net; total_assets; revenue_growth_m; operating_cash_flow_net", "thresholds": "P75; P25", "boundary": ">; <; <=", "source_alignment": "规则文档：筹资现金流持续流入但业务无明显扩张"},
    "R3-07": {"expression": "net_profit > 0 AND (operating_cash_flow_net < 0 OR ocf_profit_ratio_m < P10(ocf_profit_ratio_m))", "inputs": "net_profit; operating_cash_flow_net", "thresholds": "P10", "boundary": ">0; <0; <", "source_alignment": "实施方案 R3-07：利润为正且 CFO<0 为强异常，或 OCF/利润低于年度行业分位数"},
    "R3-08": {"expression": "neg_cfo_3y_m = 1; severity=2 when net_profit>0 OR revenue_growth_m>0, otherwise severity=1", "inputs": "operating_cash_flow_net over t,t-1,t-2; net_profit; revenue_growth_m", "thresholds": "fixed three-year condition", "boundary": "<0 for 3 consecutive years", "source_alignment": "实施方案 R3-08：连续三年 CFO 为负，利润或收入增长只提高异常程度"},
}


class RuleModelError(RuntimeError):
    pass


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    aa, bb = _num(a), _num(b)
    return aa / bb.where(bb.ne(0))


def _normalize_growth(s: pd.Series) -> pd.Series:
    x = _num(s)
    # The supplied *_yoy fields are percentage points (e.g. 8.2 means 8.2%).
    # Convert only when the distribution clearly uses that convention.
    q50, q95 = x.abs().quantile([0.50, 0.95]) if x.notna().any() else (np.nan, np.nan)
    if pd.notna(q50) and q50 > 2 and pd.notna(q95) and q95 > 5:
        x = x / 100.0
    return x


def _pct_change_by_company(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df.groupby("stock_code", sort=False)[col].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def _anon_sample_id(stock_code: str, financial_year: int) -> str:
    """Stable pseudonymous ID; never expose the real stock code downstream."""
    raw = f"{ANON_SAMPLE_SALT}|{stock_code}|{int(financial_year)}".encode("utf-8")
    return "S" + hashlib.sha256(raw).hexdigest()[:16].upper()


def _load_lockbox_spec() -> tuple[pd.DataFrame, set[str]]:
    """Load the frozen lockbox rows and the firms that must be excluded from development."""
    if not LOCKBOX_PATH.exists():
        print(f"[WARN] 锁箱文件不存在: {LOCKBOX_PATH}; 本次不生成锁箱测试样本")
        return pd.DataFrame(), set()
    try:
        all_box = pd.read_excel(LOCKBOX_PATH, sheet_name="Lockbox_All")
        required = {"stock_code", "financial_year", "sample_role"}
        missing = required - set(all_box.columns)
        if missing:
            raise RuleModelError(f"锁箱表缺少字段: {sorted(missing)}")
        all_box = all_box.copy()
        all_box["stock_code"] = all_box["stock_code"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
        all_box["financial_year"] = pd.to_numeric(all_box["financial_year"], errors="coerce").astype("Int64")
        all_box = all_box.dropna(subset=["stock_code", "financial_year"]).copy()
        all_box["financial_year"] = all_box["financial_year"].astype(int)
        all_box["lockbox_key"] = all_box["stock_code"] + "_" + all_box["financial_year"].astype(str)
        firms = set(all_box["stock_code"])
        return all_box, firms
    except Exception as exc:
        raise RuleModelError(f"无法读取冻结锁箱样本: {LOCKBOX_PATH}: {exc}") from exc


def _company_split(df: pd.DataFrame, lockbox_spec: pd.DataFrame, lockbox_firms: set[str]) -> pd.Series:
    """Return exact lockbox rows plus company-exclusive, stratified train/validation labels."""
    split = pd.Series("train", index=df.index, dtype="object")
    if not lockbox_spec.empty:
        exact_keys = set(lockbox_spec["lockbox_key"])
        row_keys = df["stock_code"] + "_" + df["financial_year"].astype(int).astype(str)
        split.loc[row_keys.isin(exact_keys)] = "lockbox"
        split.loc[df["stock_code"].isin(lockbox_firms) & split.eq("train")] = "lockbox_excluded_company_year"

    dev = df.loc[split.eq("train")].copy()
    company = dev.groupby("stock_code", as_index=False).agg(
        fraud_any=("Fraud", "max"),
        industry_parent=("industry_parent", lambda s: s.mode().iloc[0] if not s.mode().empty else "UNKNOWN"),
    )
    rng = np.random.default_rng(SPLIT_SEED)
    validation_firms: set[str] = set()
    for _, bucket in company.groupby(["fraud_any", "industry_parent"], dropna=False):
        n_validation = int(math.floor(len(bucket) * (1.0 - TRAIN_SHARE)))
        if len(bucket) >= 5:
            n_validation = max(1, n_validation)
        if n_validation:
            chosen = rng.choice(bucket["stock_code"].to_numpy(), size=min(n_validation, len(bucket)), replace=False)
            validation_firms.update(chosen.tolist())
    split.loc[split.eq("train") & df["stock_code"].isin(validation_firms)] = "validation"
    return split


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path, low_memory=False)
    required = {"stock_code", "year", "industry_code", "total_assets", "operating_revenue", "net_profit", "operating_cash_flow_net"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuleModelError(f"汇总数据缺少必要字段: {missing}")

    df["stock_code"] = df["stock_code"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    df["financial_year"] = pd.to_datetime(df["year"], errors="coerce").dt.year
    df = df[df["financial_year"].between(2007, 2025)].copy()

    # The supplied file is intended to exclude finance, but a few inconsistent
    # labels remain. Remove them explicitly and report the count in metadata.
    fin_text = (df.get("industry_name", pd.Series("", index=df.index)).fillna("").astype(str) + " " +
                df.get("Indnme", pd.Series("", index=df.index)).fillna("").astype(str))
    finance_mask = fin_text.str.contains("银行|保险|证券|信托|金融", regex=True)
    excluded_finance = int(finance_mask.sum())
    df = df.loc[~finance_mask].copy()

    text_cols = {"stock_code", "company_name", "industry_code", "industry_name", "Indcd", "Indnme", "year", "fraud_type"}
    for c in df.columns:
        if c not in text_cols:
            df[c] = _num(df[c])
    df["Fraud"] = _num(df.get("Fraud", pd.Series(0, index=df.index))).fillna(0).astype(int)
    df["industry_code"] = df["industry_code"].fillna(df.get("Indcd", pd.Series("UNKNOWN", index=df.index))).astype(str)
    df["industry_code"] = df["industry_code"].replace({"nan": "UNKNOWN", "None": "UNKNOWN"})
    df["industry_parent"] = df.get("Indcd", pd.Series("UNKNOWN", index=df.index)).fillna("UNKNOWN").astype(str)
    df["industry_parent"] = df["industry_parent"].replace({"nan": "UNKNOWN", "None": "UNKNOWN"})
    df = df.sort_values(["stock_code", "financial_year"]).reset_index(drop=True)

    # Normalized raw metrics. Every metric used by a rule has a named column.
    df["revenue_growth_m"] = _normalize_growth(df.get("revenue_growth_yoy", pd.Series(np.nan, index=df.index)))
    if "revenue_growth" in df:
        df["revenue_growth_m"] = df["revenue_growth_m"].fillna(_normalize_growth(df["revenue_growth"]))
    df["revenue_growth_m"] = df["revenue_growth_m"].fillna(_pct_change_by_company(df, "operating_revenue"))

    for out, src in [("receivable_growth_m", "receivable_growth_yoy"), ("inventory_growth_m", "inventory_growth_yoy"),
                     ("cost_growth_m", "cost_of_revenue_growth_yoy")]:
        df[out] = _normalize_growth(df.get(src, pd.Series(np.nan, index=df.index)))
    df["receivable_growth_m"] = df["receivable_growth_m"].fillna(_pct_change_by_company(df, "accounts_receivable_net"))
    df["inventory_growth_m"] = df["inventory_growth_m"].fillna(_pct_change_by_company(df, "inventory_net"))
    df["cost_growth_m"] = df["cost_growth_m"].fillna(_pct_change_by_company(df, "operating_cost"))

    df["gross_margin_m"] = _num(df.get("gross_profit_margin", pd.Series(np.nan, index=df.index)))
    df["gross_margin_m"] = df["gross_margin_m"].fillna(_safe_div(df["operating_revenue"] - df["operating_cost"], df["operating_revenue"]))
    df["sales_expense_ratio_m"] = _num(df.get("selling_expense_ratio", pd.Series(np.nan, index=df.index)))
    df["sales_expense_ratio_m"] = df["sales_expense_ratio_m"].where(df["sales_expense_ratio_m"].abs().le(5))
    df["sales_expense_ratio_m"] = df["sales_expense_ratio_m"].fillna(_safe_div(df["selling_expenses"], df["operating_revenue"]))
    df["sales_expense_drop_m"] = df.groupby("stock_code", sort=False)["sales_expense_ratio_m"].shift(1) - df["sales_expense_ratio_m"]
    df["net_profit_growth_m"] = _pct_change_by_company(df, "net_profit")
    df["cash_collection_ratio_m"] = _safe_div(df["cash_received_from_sales"], df["operating_revenue"])
    df["ocf_profit_ratio_m"] = _safe_div(df["operating_cash_flow_net"], df["net_profit"])
    df["inventory_to_assets_m"] = _safe_div(df["inventory_net"], df["total_assets"])
    df["inventory_turnover_m"] = _num(df.get("inventory_turnover_ratio", pd.Series(np.nan, index=df.index)))
    df["inventory_turnover_m"] = df["inventory_turnover_m"].fillna(_safe_div(df["operating_cost"], df["inventory_net"]))
    df["construction_growth_m"] = _pct_change_by_company(df, "construction_in_progress_net")
    df["fixed_assets_growth_m"] = _pct_change_by_company(df, "fixed_assets_net")
    df["construction_growth_lag_m"] = df.groupby("stock_code", sort=False)["construction_growth_m"].shift(1)
    df["cash_to_assets_m"] = _safe_div(df["cash"], df["total_assets"])
    df["short_loan_to_liab_m"] = _safe_div(df["short_term_borrowings"], df["total_liabilities"])
    df["prepayment_to_assets_m"] = _safe_div(df["prepayments_net"], df["total_assets"])
    df["prepayment_growth_m"] = _pct_change_by_company(df, "prepayments_net")
    df["related_party_to_assets_m"] = _safe_div(df["related_party_transaction_amount"], df["total_assets"])
    df["interest_income_to_cash_m"] = _safe_div(df["interest_income"], df["cash"])
    df["financing_to_assets_m"] = _safe_div(df["financing_cash_flow_net"], df["total_assets"])
    cfo = _num(df["operating_cash_flow_net"])
    neg = cfo.notna() & cfo.lt(0)
    prev_year = df.groupby("stock_code", sort=False)["financial_year"].shift(1)
    prev2_year = df.groupby("stock_code", sort=False)["financial_year"].shift(2)
    consecutive = df["financial_year"].sub(prev_year).eq(1) & prev_year.sub(prev2_year).eq(1)
    neg1 = neg.groupby(df["stock_code"], sort=False).shift(1)
    neg2 = neg.groupby(df["stock_code"], sort=False).shift(2)
    eligible = cfo.notna() & neg1.notna() & neg2.notna() & consecutive
    df["neg_cfo_3y_m"] = pd.Series(np.nan, index=df.index, dtype="float64")
    df.loc[eligible, "neg_cfo_3y_m"] = (neg & neg1 & neg2).loc[eligible].astype(float)

    # Keep the real company-year key only in-memory for the internal mapping
    # sidecar.  RulePacket and model-facing files receive the pseudonymous ID.
    df["internal_sample_key"] = df["stock_code"] + "_" + df["financial_year"].astype(int).astype(str)
    df["sample_id"] = [
        _anon_sample_id(code, year)
        for code, year in zip(df["stock_code"], df["financial_year"])
    ]
    return df, {"excluded_finance_rows": excluded_finance, "rows": int(len(df)), "year_min": int(df.financial_year.min()), "year_max": int(df.financial_year.max())}


METRICS = [
    "revenue_growth_m", "receivable_growth_m", "inventory_growth_m", "cost_growth_m", "gross_margin_m",
    "sales_expense_ratio_m", "sales_expense_drop_m", "net_profit_growth_m", "cash_collection_ratio_m",
    "ocf_profit_ratio_m", "inventory_to_assets_m", "inventory_turnover_m", "construction_growth_m",
    "construction_growth_lag_m", "fixed_assets_growth_m", "cash_to_assets_m", "short_loan_to_liab_m",
    "prepayment_to_assets_m", "prepayment_growth_m", "related_party_to_assets_m", "interest_income_to_cash_m",
    "financing_to_assets_m", "receivable_minus_revenue_growth_m", "inventory_minus_cost_growth_m",
]


def _group_stats(sub: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, part in sub.groupby(key_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(key_cols, keys))
        for metric in METRICS:
            x = _num(part[metric]).dropna()
            if x.empty:
                continue
            rows.append({
                **base, "metric_name": metric, "n_obs": int(x.size),
                "mean": float(x.mean()), "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
                "median": float(x.quantile(.50)), "p10": float(x.quantile(.10)),
                "p25": float(x.quantile(.25)), "p75": float(x.quantile(.75)), "p90": float(x.quantile(.90)),
            })
    return pd.DataFrame(rows)


def build_threshold_table(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Build frozen same-year thresholds with detailed -> parent -> market fallback."""
    train = df.loc[train_mask].copy()
    detail = _group_stats(train, ["financial_year", "industry_code"])
    parent = _group_stats(train, ["financial_year", "industry_parent"])
    market = _group_stats(train, ["financial_year"])
    detail["industry_level"] = "subindustry"
    detail["group_code"] = detail["industry_code"]
    detail["threshold_source"] = "subindustry_year"
    parent["industry_level"] = "level1"
    parent["group_code"] = parent["industry_parent"]
    parent["threshold_source"] = "level1_year"
    market["industry_level"] = "market"
    market["group_code"] = "ALL"
    market["threshold_source"] = "market_year"
    for x in (detail, parent, market):
        x["threshold_version"] = THRESHOLD_VERSION
        x["n_min"] = N_MIN
        x["threshold_available"] = x["n_obs"].ge(N_MIN)
    out = pd.concat([detail, parent, market], ignore_index=True)
    cols = ["financial_year", "industry_level", "group_code", "industry_code", "industry_parent", "metric_name", "n_obs", "n_min",
            "mean", "std", "median", "p10", "p25", "p75", "p90", "threshold_source", "threshold_version", "threshold_available"]
    return out[cols].sort_values(["financial_year", "metric_name", "industry_level", "group_code"]).reset_index(drop=True)


@dataclass
class StatLookup:
    threshold_table: pd.DataFrame

    def __post_init__(self) -> None:
        self._idx = self.threshold_table.set_index(["financial_year", "industry_level", "group_code", "metric_name"])

    def get(self, year: int, detail: str, parent: str, metric: str) -> dict[str, Any] | None:
        for level, code in (("subindustry", detail), ("level1", parent), ("market", "ALL")):
            try:
                r = self._idx.loc[(year, level, code, metric)]
            except KeyError:
                continue
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            if bool(r["threshold_available"]):
                return {"industry_level": level, "group_code": code, **r.to_dict()}
        return None


def _stat_distance(value: float | None, threshold: float | None, std: float | None, direction: str) -> float | None:
    if value is None or threshold is None or pd.isna(value) or pd.isna(threshold):
        return None
    scale = float(std) if std is not None and pd.notna(std) and float(std) > 1e-12 else max(abs(float(threshold)), 1.0)
    d = (float(value) - float(threshold)) / scale
    return float(np.clip(d if direction == "high" else -d, -10, 10))


def _f(v: Any) -> float | None:
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def _one_rule(row: pd.Series, lookup: StatLookup, code: str) -> dict[str, Any]:
    year, detail, parent = int(row.financial_year), str(row.industry_code), str(row.industry_parent)
    d = row.to_dict()
    missing: list[str] = []
    thresholds: dict[str, Any] = {}
    distances: dict[str, float] = {}

    def metric(name: str) -> float | None:
        v = _f(d.get(name))
        if v is None:
            missing.append(name)
        return v

    def q(name: str, p: str) -> float | None:
        st = lookup.get(year, detail, parent, name)
        if st is None:
            missing.append(f"threshold:{name}")
            return None
        value_key = "median" if p == "p50" else p
        thresholds[f"{name}:{p}"] = {"metric": name, "quantile": p, "value": _f(st.get(value_key)), "source": st.get("threshold_source"), "level": st.get("industry_level"), "n_obs": int(st.get("n_obs", 0))}
        return _f(st.get(value_key))

    def dist(name: str, value: float | None, p: str, direction: str) -> None:
        st = lookup.get(year, detail, parent, name)
        if st is not None:
            x = _stat_distance(value, _f(st.get(p)), _f(st.get("std")), direction)
            if x is not None:
                distances[f"{name}:{p}"] = x

    hit: bool | None = None
    raw: dict[str, Any] = {}
    try:
        if code == "R1-01":
            a, b = metric("receivable_growth_m"), metric("revenue_growth_m"); x = None if a is None or b is None else a - b
            # Gap threshold is computed separately below because it is a derived metric.
            gap = _f(d.get("receivable_minus_revenue_growth_m")); th = q("receivable_minus_revenue_growth_m", "p75")
            raw = {"gap": gap, "receivable_growth": a, "revenue_growth": b}; hit = None if gap is None or th is None else gap > th; dist("receivable_minus_revenue_growth_m", gap, "p75", "high")
        elif code == "R1-02":
            x, g = metric("cash_collection_ratio_m"), metric("revenue_growth_m"); t1, t2 = q("cash_collection_ratio_m", "p25"), q("revenue_growth_m", "p50")
            raw = {"cash_collection_ratio": x, "revenue_growth": g}; hit = None if None in (x, g, t1, t2) else x < t1 and g > t2; dist("cash_collection_ratio_m", x, "p25", "low"); dist("revenue_growth_m", g, "p50", "high")
        elif code == "R1-03":
            x, t = metric("gross_margin_m"), q("gross_margin_m", "p90"); raw = {"gross_margin": x}; hit = None if x is None or t is None else x > t; dist("gross_margin_m", x, "p90", "high")
        elif code == "R1-04":
            g, x = metric("net_profit_growth_m"), metric("ocf_profit_ratio_m"); t1, t2 = q("net_profit_growth_m", "p75"), q("ocf_profit_ratio_m", "p25"); raw = {"net_profit_growth": g, "ocf_profit_ratio": x}; hit = None if None in (g, x, t1, t2) else g > t1 and x < t2; dist("net_profit_growth_m", g, "p75", "high"); dist("ocf_profit_ratio_m", x, "p25", "low")
        elif code == "R1-05":
            x, t = metric("revenue_growth_m"), q("revenue_growth_m", "p90"); raw = {"revenue_growth": x}; hit = None if x is None or t is None else x > t; dist("revenue_growth_m", x, "p90", "high")
        elif code == "R1-06":
            g, drop = metric("revenue_growth_m"), metric("sales_expense_drop_m"); t1, t2 = q("revenue_growth_m", "p75"), q("sales_expense_drop_m", "p75"); raw = {"revenue_growth": g, "sales_expense_drop": drop}; hit = None if None in (g, drop, t1, t2) else g > t1 and drop > t2; dist("revenue_growth_m", g, "p75", "high"); dist("sales_expense_drop_m", drop, "p75", "high")
        elif code == "R2-01":
            a, b = metric("inventory_growth_m"), metric("cost_growth_m"); gap = None if a is None or b is None else a - b; t = q("inventory_minus_cost_growth_m", "p75"); raw = {"inventory_growth": a, "cost_growth": b, "gap": gap}; hit = None if gap is None or t is None else gap > t; dist("inventory_minus_cost_growth_m", gap, "p75", "high")
        elif code == "R2-02":
            x, t = metric("inventory_turnover_m"), q("inventory_turnover_m", "p25"); raw = {"inventory_turnover": x}; hit = None if x is None or t is None else x < t; dist("inventory_turnover_m", x, "p25", "low")
        elif code == "R2-03":
            a, b = metric("inventory_to_assets_m"), metric("inventory_turnover_m"); t1, t2 = q("inventory_to_assets_m", "p75"), q("inventory_turnover_m", "p25"); raw = {"inventory_to_assets": a, "inventory_turnover": b}; hit = None if None in (a, b, t1, t2) else a > t1 and b < t2; dist("inventory_to_assets_m", a, "p75", "high"); dist("inventory_turnover_m", b, "p25", "low")
        elif code == "R2-04":
            x, t = metric("inventory_to_assets_m"), q("inventory_to_assets_m", "p90"); raw = {"inventory_to_assets": x}; hit = None if x is None or t is None else x > t; dist("inventory_to_assets_m", x, "p90", "high")
        elif code == "R2-05":
            a, b, c = metric("construction_growth_m"), metric("construction_growth_lag_m"), metric("fixed_assets_growth_m"); t1, t2 = q("construction_growth_m", "p75"), q("fixed_assets_growth_m", "p25"); raw = {"construction_growth": a, "construction_growth_lag": b, "fixed_assets_growth": c}; hit = None if None in (a, b, c, t1, t2) else a > t1 and b > t1 and c < t2; dist("construction_growth_m", a, "p75", "high"); dist("fixed_assets_growth_m", c, "p25", "low")
        elif code == "R3-01":
            a, b = metric("cash_to_assets_m"), metric("short_loan_to_liab_m"); t1, t2 = q("cash_to_assets_m", "p75"), q("short_loan_to_liab_m", "p75"); raw = {"cash_to_assets": a, "short_loan_to_liab": b}; hit = None if None in (a, b, t1, t2) else a > t1 and b > t2; dist("cash_to_assets_m", a, "p75", "high"); dist("short_loan_to_liab_m", b, "p75", "high")
        elif code == "R3-02":
            a, b = metric("prepayment_to_assets_m"), metric("prepayment_growth_m")
            t1, t2 = q("prepayment_to_assets_m", "p75"), q("prepayment_growth_m", "p75")
            raw = {"prepayment_to_assets": a, "prepayment_growth": b}
            a_available, b_available = None not in (a, t1), None not in (b, t2)
            if a_available and a > t1 or b_available and b > t2:
                hit = True
                # For an OR rule, one satisfied branch is sufficient.
                missing.clear()
            elif a_available and b_available:
                hit = False
                missing.clear()
            else:
                hit = None
            dist("prepayment_to_assets_m", a, "p75", "high"); dist("prepayment_growth_m", b, "p75", "high")
        elif code == "R3-03":
            p, x = metric("net_profit"), metric("ocf_profit_ratio_m"); t = q("ocf_profit_ratio_m", "p25"); raw = {"net_profit": p, "ocf_profit_ratio": x}; hit = None if None in (p, x, t) else p > 0 and x < t; dist("ocf_profit_ratio_m", x, "p25", "low")
        elif code == "R3-04":
            x, t = metric("related_party_to_assets_m"), q("related_party_to_assets_m", "p90"); raw = {"related_party_to_assets": x}; hit = None if x is None or t is None else x > t; dist("related_party_to_assets_m", x, "p90", "high")
        elif code == "R3-05":
            a, b = metric("cash_to_assets_m"), metric("interest_income_to_cash_m"); t1, t2 = q("cash_to_assets_m", "p75"), q("interest_income_to_cash_m", "p25"); raw = {"cash_to_assets": a, "interest_income_to_cash": b}; hit = None if None in (a, b, t1, t2) else a > t1 and b < t2; dist("cash_to_assets_m", a, "p75", "high"); dist("interest_income_to_cash_m", b, "p25", "low")
        elif code == "R3-06":
            f, g, c = metric("financing_to_assets_m"), metric("revenue_growth_m"), metric("operating_cash_flow_net"); t1, t2 = q("financing_to_assets_m", "p75"), q("revenue_growth_m", "p25"); raw = {"financing_to_assets": f, "revenue_growth": g, "operating_cash_flow": c}; hit = None if None in (f, g, c, t1, t2) else f > t1 and g < t2 and c <= 0; dist("financing_to_assets_m", f, "p75", "high"); dist("revenue_growth_m", g, "p25", "low")
        elif code == "R3-07":
            p = metric("net_profit")
            c = _f(d.get("operating_cash_flow_net"))
            x = _f(d.get("ocf_profit_ratio_m"))
            t = q("ocf_profit_ratio_m", "p10")
            raw = {"net_profit": p, "operating_cash_flow": c, "ocf_profit_ratio": x}
            if p is None:
                hit = None
            elif p <= 0:
                hit = False
                missing.clear()
            elif c is not None and c < 0:
                hit = True
                missing.clear()
            elif x is not None and t is not None:
                hit = x < t
                missing.clear()
            else:
                hit = None
            dist("ocf_profit_ratio_m", x, "p10", "low")
        elif code == "R3-08":
            # The implementation plan defines the three-year negative CFO as
            # the hit condition.  Current profit or revenue growth only
            # increases severity and is therefore optional context, not a
            # prerequisite for calculating the rule.
            n = metric("neg_cfo_3y_m")
            p, g = _f(d.get("net_profit")), _f(d.get("revenue_growth_m"))
            raw = {"neg_cfo_3y": n, "net_profit": p, "revenue_growth": g}
            hit = None if n is None else n >= 1
        else:
            raise RuleModelError(f"未定义规则: {code}")
    except Exception as exc:
        return _result(row, code, None, raw, thresholds, distances, [f"calculation:{exc}"], "calculation_error")

    if missing:
        return _result(row, code, None, raw, thresholds, distances, sorted(set(missing)), "data_insufficient")
    # Keep status semantics separate from the binary rule_hit field.  The
    # implementation plan reserves `watch` for an abnormal/watch condition;
    # an unavailable rule is never silently converted to normal.
    status = "watch" if bool(hit) else "normal"
    return _result(row, code, bool(hit), raw, thresholds, distances, [], status)


def _result(row: pd.Series, code: str, hit: bool | None, raw: dict[str, Any], thresholds: dict[str, Any], distances: dict[str, float], missing: list[str], status: str) -> dict[str, Any]:
    if hit is None:
        severity = None
    elif not hit:
        severity = 0
    elif code == "R3-08":
        # R3-08 is a hit whenever CFO is negative for three consecutive years.
        # Profit or positive growth upgrades the signal from general to strong.
        severity = 2 if (raw.get("net_profit", 0) or 0) > 0 or (raw.get("revenue_growth", 0) or 0) > 0 else 1
    elif code == "R3-07":
        severity = 2
    else:
        severity = 1
    ev = f"EV-{row.sample_id}-{code}"
    return {
        "rule_id": code, "rule_name": RULE_NAMES[code], "category": RULE_CATEGORY[code],
        "rule_hit": None if hit is None else int(bool(hit)), "severity": severity, "raw_value": raw,
        "threshold_values": thresholds, "threshold_source": sorted({v.get("source") for v in thresholds.values() if v.get("source")}),
        "condition_distances": distances,
        "distance_to_threshold": float(min(distances.values())) if distances else None,
        "evidence_id": ev, "rule_status": status, "missing_fields": missing,
    }


def _derived_threshold_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["receivable_minus_revenue_growth_m"] = df["receivable_growth_m"] - df["revenue_growth_m"]
    df["inventory_minus_cost_growth_m"] = df["inventory_growth_m"] - df["cost_growth_m"]
    return df


def _evidence_text(rule: dict[str, Any]) -> str:
    raw = json.dumps(rule.get("raw_value", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    th = json.dumps(rule.get("threshold_values", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{rule['rule_name']}；raw={raw}；threshold={th}；status={rule['rule_status']}。"


def _formula_crosswalk_rows() -> list[dict[str, Any]]:
    rows = []
    for code, category, name, logic in RULE_CATALOG:
        frozen = RULE_FORMULA_FREEZE[code]
        rows.append({
            "rule_id": code,
            "category": category,
            "rule_name": name,
            "business_logic_from_rule_doc": logic,
            "frozen_expression": frozen["expression"],
            "required_inputs": frozen["inputs"],
            "thresholds_or_fixed_condition": frozen["thresholds"],
            "boundary": frozen["boundary"],
            "source_alignment": frozen["source_alignment"],
            "formula_version": RULE_FORMULA_VERSION,
            "review_status": "frozen_for_rule_engine_v1",
        })
    return rows


def generate_outputs(df: pd.DataFrame, thresholds: pd.DataFrame, split: pd.Series, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    lookup = StatLookup(thresholds)
    catalog_rows = []
    for code, cat, name, logic in RULE_CATALOG:
        catalog_rows.append({"rule_id": code, "category": cat, "rule_name": name, "logic": logic, "fixed_business_conditions": "仅使用实施方案允许的固定条件；统计阈值由行业×财务年度表提供", "formula_version": RULE_FORMULA_VERSION, "version": VERSION})
    pd.DataFrame(catalog_rows).to_csv(out_dir / "rule_catalog.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(_formula_crosswalk_rows()).to_csv(out_dir / "rule_formula_crosswalk.csv", index=False, encoding="utf-8-sig")
    (out_dir / "rule_formula_freeze.json").write_text(json.dumps({
        "formula_version": RULE_FORMULA_VERSION,
        "rule_engine_version": VERSION,
        "basis": "规则.docx业务意图 + 实施方案行业×财务年度阈值和R3-07/R3-08说明",
        "review_status": "frozen_for_rule_engine_v1",
        "rules": RULE_FORMULA_FREEZE,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    thresholds.to_csv(out_dir / "threshold_table.csv", index=False, encoding="utf-8-sig")

    feature_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    packet_path = out_dir / "rule_packets.jsonl"
    with packet_path.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            rules = [_one_rule(row, lookup, code) for code, *_ in RULE_CATALOG]
            for r in rules:
                r["evidence_text"] = _evidence_text(r)
            packet = {
                "rule_packet_version": VERSION, "threshold_version": THRESHOLD_VERSION,
                "sample_id": row.sample_id, "financial_year": int(row.financial_year),
                "industry_code": row.industry_code, "industry_parent": row.industry_parent,
                "rules": rules,
            }
            canonical = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            packet["rule_packet_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            fh.write(json.dumps(packet, ensure_ascii=False, separators=(",", ":")) + "\n")

            fr = {"sample_id": row.sample_id, "financial_year": int(row.financial_year), "industry_code": row.industry_code,
                  "industry_parent": row.industry_parent,
                  "rule_packet_hash": packet["rule_packet_hash"]}
            for r in rules:
                tag = r["rule_id"].replace("-", "_")
                fr[f"{tag}_hit"] = r["rule_hit"]
                fr[f"{tag}_severity"] = r["severity"]
                fr[f"{tag}_distance"] = r["distance_to_threshold"]
                fr[f"{tag}_available"] = 0 if r["rule_status"] in {"data_insufficient", "calculation_error"} else 1
            feature_rows.append(fr)
            evaluation_rows.append({
                "sample_id": row.sample_id,
                "financial_year": int(row.financial_year),
                "fraud_label": int(row.Fraud) if "Fraud" in row and pd.notna(row.Fraud) else None,
                "fraud_type": row.get("fraud_type") if "fraud_type" in row else None,
                "split": split.loc[row.name],
            })
            mapping_rows.append({
                "sample_id": row.sample_id,
                "internal_sample_key": row.internal_sample_key,
                "stock_code": row.stock_code,
                "financial_year": int(row.financial_year),
                "split": split.loc[row.name],
                "sample_role": row.get("lockbox_sample_role"),
                "pair_id": row.get("lockbox_pair_id"),
                "match_rank": row.get("lockbox_match_rank"),
                "match_level": row.get("lockbox_match_level"),
            })
    pd.DataFrame(feature_rows).to_csv(out_dir / "rule_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(evaluation_rows).to_csv(out_dir / "evaluation_labels.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mapping_rows).to_csv(out_dir / "sample_id_map_internal.csv", index=False, encoding="utf-8-sig")
    # Model-facing ML view.  No labels, split markers, stock codes, or company
    # names are present; downstream code should read this file explicitly.
    ml_cols = ["sample_id"]
    for code, *_ in RULE_CATALOG:
        tag = code.replace("-", "_")
        ml_cols.extend([f"{tag}_hit", f"{tag}_severity", f"{tag}_distance", f"{tag}_available"])
    pd.DataFrame(feature_rows)[ml_cols].to_csv(out_dir / "ml_features.csv", index=False, encoding="utf-8-sig")

    # Compact diagnostics for audit and downstream reporting.
    feature_df = pd.DataFrame(feature_rows)
    diag = []
    for code, *_ in RULE_CATALOG:
        tag = code.replace("-", "_")
        hit = feature_df[f"{tag}_hit"]
        avail = feature_df[f"{tag}_available"]
        diag.append({"rule_id": code, "n_samples": len(feature_df), "n_available": int(avail.sum()), "availability_rate": float(avail.mean()),
                     "n_hit": int((hit == 1).sum()), "hit_rate_on_available": float((hit[hit.notna()] == 1).mean()) if hit.notna().any() else None})
    pd.DataFrame(diag).to_csv(out_dir / "rule_diagnostics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "rule_engine_version": VERSION, "threshold_version": THRESHOLD_VERSION, "n_min": N_MIN,
        "split_seed": SPLIT_SEED, "train_share": TRAIN_SHARE, "rows": int(len(df)),
        "companies": int(df.stock_code.nunique()), "years": [int(df.financial_year.min()), int(df.financial_year.max())],
        "split_counts": split.value_counts().to_dict(), "fraud_counts": pd.to_numeric(df.get("Fraud", 0), errors="coerce").fillna(0).astype(int).value_counts().to_dict(),
        "threshold_table_rows": int(len(thresholds)),
        "threshold_available_rows": int(thresholds["threshold_available"].sum()),
        "threshold_unavailable_rows": int((~thresholds["threshold_available"]).sum()),
        "rule_formula_version": RULE_FORMULA_VERSION,
        "anon_sample_version": ANON_SAMPLE_VERSION,
        "rule_packet_contract": {"ml_allowed": ["sample_id", "rule_hit", "severity", "distance_to_threshold", "available"], "agent_allowed": ["sample_id", "financial_year", "industry_code", "industry_parent", "rule_hit", "severity", "raw_value", "threshold_values", "distance_to_threshold", "evidence_id", "evidence_text", "rule_status"], "forbidden_to_ml_and_agent": ["fraud_label", "split", "stock_code", "company_name", "MLProbability", "ML_RiskLevel", "feature_importances", "SHAP", "Agent_RiskLevel"]},
    }
    (out_dir / "model_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    df, meta = load_data()
    lockbox_spec, lockbox_firms = _load_lockbox_spec()
    split = _company_split(df, lockbox_spec, lockbox_firms)
    df["split"] = split
    df["lockbox_sample_role"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df["lockbox_pair_id"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df["lockbox_match_rank"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df["lockbox_match_level"] = pd.Series(pd.NA, index=df.index, dtype="object")
    if not lockbox_spec.empty:
        spec = lockbox_spec.set_index("lockbox_key")
        keys = df["stock_code"] + "_" + df["financial_year"].astype(int).astype(str)
        for source, target in [("sample_role", "lockbox_sample_role"), ("pair_id", "lockbox_pair_id"), ("match_rank", "lockbox_match_rank"), ("match_level", "lockbox_match_level")]:
            df[target] = keys.map(spec[source])
    df = _derived_threshold_metrics(df)
    train_mask = split.eq("train")
    thresholds = build_threshold_table(df, train_mask)
    summary = generate_outputs(df, thresholds, split)
    summary.update(meta)
    summary["lockbox_firms"] = len(lockbox_firms)
    summary["lockbox_test_rows"] = int(split.eq("lockbox").sum())
    summary["lockbox_excluded_company_year_rows"] = int(split.eq("lockbox_excluded_company_year").sum())
    (OUT_DIR / "model_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
