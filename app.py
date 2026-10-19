#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask 后端 API
提供股票查询、工作底稿生成、三智能体分析接口
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import sys
import json
import pandas as pd
from pathlib import Path

# 确保项目目录在路径中
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

app = Flask(__name__, static_folder=str(SCRIPT_DIR), static_url_path='')
CORS(app)

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(Path(SCRIPT_DIR) / '.env')

DATA_PATH = Path(SCRIPT_DIR) / '汇总数据.csv'

# ── 全局缓存 ──
_df_raw = None
_workpaper_gen = None
_ml_reference_cache = {}


def _json_fallback(value):
    """将 numpy/pandas 标量安全转换为标准 JSON 可序列化值。"""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (ValueError, TypeError):
            pass
    return str(value)


def _json_safe(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=_json_fallback))


def get_df():
    global _df_raw
    if _df_raw is None:
        _df_raw = pd.read_csv(DATA_PATH, encoding='utf-8')
        _df_raw['stock_code'] = _df_raw['stock_code'].astype(str).str.zfill(6)
        _df_raw['year_num'] = pd.to_datetime(_df_raw['year'], errors='coerce').dt.year
    return _df_raw


def get_workpaper_generator():
    global _workpaper_gen
    if _workpaper_gen is None:
        import workpaper_generator as wg
        _workpaper_gen = wg.WorkpaperGenerator()
    return _workpaper_gen


# ═══════════════════════════════════════════════════════════════
#  API 路由R
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_file(str(SCRIPT_DIR / 'index.html'))


@app.route('/api/stock/search')
def stock_search():
    """搜索股票：支持按代码或名称模糊搜索"""
    keyword = request.args.get('q', '').strip()
    if not keyword:
        return jsonify({"error": "请输入搜索关键词"}), 400

    df = get_df()
    mask = (
        df['stock_code'].str.contains(keyword) |
        df['company_name'].str.contains(keyword, na=False)
    )
    results = df[mask][['stock_code', 'company_name', 'industry_name']].drop_duplicates()
    results['industry_name'] = results['industry_name'].fillna('').astype(str)
    results = results.head(20)
    return jsonify(results.to_dict('records'))


@app.route('/api/stock/info')
def stock_info():
    """获取指定股票的可用年份列表和基本信息"""
    stock_code = request.args.get('code', '').strip().zfill(6)
    if not stock_code:
        return jsonify({"error": "请输入股票代码"}), 400

    df = get_df()
    mask = df['stock_code'] == stock_code
    if mask.sum() == 0:
        return jsonify({"error": f"未找到股票代码 {stock_code}"}), 404

    company = df[mask].iloc[0]
    years = sorted(df[mask]['year_num'].dropna().unique().tolist(), reverse=True)

    return jsonify({
        "stock_code": stock_code,
        "company_name": str(company['company_name']),
        "industry_name": str(company.get('industry_name', '')),
        "available_years": years,
    })


@app.route('/api/stock/basic_info')
def stock_basic_info():
    """获取指定股票某年的详细基本信息和财务数据"""
    stock_code = request.args.get('code', '').strip().zfill(6)
    year = request.args.get('year', '').strip()

    if not stock_code or not year:
        return jsonify({"error": "请输入股票代码和年份"}), 400

    df = get_df()
    year_int = int(year)
    mask = (df['stock_code'] == stock_code) & (df['year_num'] == year_int)
    if mask.sum() == 0:
        return jsonify({"error": f"未找到 {stock_code} {year} 年数据"}), 404

    row = df[mask].iloc[0]

    def fmt_b(v):
        if pd.isna(v) or v is None:
            return None
        return round(float(v) / 1e8, 2)

    def fmt_pct(v):
        if pd.isna(v) or v is None:
            return None
        return round(float(v) * 100, 2)

    info = {
        "stock_code": stock_code,
        "company_name": str(row['company_name']),
        "industry_name": str(row.get('industry_name', '')),
        "industry_code": str(row.get('industry_code', '')),
        "year": year_int,
        "report_date": str(row.get('year', '')),
        # 规模指标（亿元）
        "total_assets": fmt_b(row.get('total_assets')),
        "total_revenue": fmt_b(row.get('operating_revenue')),
        "net_profit": fmt_b(row.get('net_profit')),
        "total_liabilities": fmt_b(row.get('total_liabilities')),
        "owners_equity": fmt_b(row.get('total_owners_equity')),
        "operating_cash_flow": fmt_b(row.get('operating_cash_flow_net')),
        # 盈利能力
        "gross_margin": fmt_pct(row.get('gross_profit_margin')),
        "net_profit_margin": fmt_pct(row.get('net_profit') / row.get('total_operating_revenue')) if row.get('total_operating_revenue') and row.get('total_operating_revenue') != 0 else None,
        "roe": float(row['weighted_avg_roe']) if pd.notna(row.get('weighted_avg_roe')) else None,
        "basic_eps": float(row['basic_eps']) if pd.notna(row.get('basic_eps')) else None,
        # 营运能力
        "receivables_turnover": float(row['receivables_turnover_ratio']) if pd.notna(row.get('receivables_turnover_ratio')) else None,
        "inventory_turnover": float(row['inventory_turnover_ratio']) if pd.notna(row.get('inventory_turnover_ratio')) else None,
        "current_ratio": float(row['CurrentRatio']) if pd.notna(row.get('CurrentRatio')) else None,
        # 增长率
        "revenue_growth": fmt_pct(row.get('revenue_growth')),
        "receivable_growth": fmt_pct(row.get('receivable_growth_yoy')),
        "inventory_growth": fmt_pct(row.get('inventory_growth_yoy')),
        # 现金流
        "ocf": fmt_b(row.get('operating_cash_flow_net')),
        "investing_cf": fmt_b(row.get('investing_cash_flow_net')),
        "financing_cf": fmt_b(row.get('financing_cash_flow_net')),
        # 非财务
        "audit_opinion": str(row.get('audit_opinion_risk', '')),
        "customer_concentration": float(row['customer_concentration']) if pd.notna(row.get('customer_concentration')) else None,
        "supplier_concentration": float(row['supplier_concentration']) if pd.notna(row.get('supplier_concentration')) else None,
        "related_party_amount": float(row['related_party_transaction_amount']) if pd.notna(row.get('related_party_transaction_amount')) else None,
        # 费用率
        "selling_expense_ratio": fmt_pct(row.get('selling_expense_ratio')),
        "interest_liab_ratio": float(row['interest_liab_ratio']) if pd.notna(row.get('interest_liab_ratio')) else None,
    }

    return jsonify(info)


@app.route('/api/workpaper')
def generate_workpaper():
    """生成工作底稿（不调用LLM）"""
    stock_code = request.args.get('code', '').strip().zfill(6)
    year = request.args.get('year', '').strip()

    if not stock_code or not year:
        return jsonify({"error": "请输入股票代码和年份"}), 400

    try:
        gen = get_workpaper_generator()
        wp = gen.generate(stock_code, int(year))
        if not wp.get('success'):
            return jsonify({"error": wp.get('error', '底稿生成失败')}), 500
        return jsonify(_json_safe(wp))
    except Exception as e:
        return jsonify({"error": f"底稿生成失败: {str(e)}"}), 500


@app.route('/api/ml')
def ml_reference():
    """读取冻结模型的单样本预测结果（不在接口请求中重新训练）。"""
    stock_code = request.args.get('code', '').strip().zfill(6)
    year = request.args.get('year', '').strip()
    if not stock_code or not year:
        return jsonify({"error": "请输入股票代码和年份"}), 400

    try:
        year_int = int(year)
    except (TypeError, ValueError):
        return jsonify({"error": "年份格式无效"}), 400

    cache_key = (stock_code, year_int)
    if cache_key in _ml_reference_cache:
        return jsonify(_json_safe(_ml_reference_cache[cache_key]))

    try:
        import three_agents_three_level as ta

        reference = ta.load_frozen_ml_reference(stock_code, year_int)
        metadata_path = SCRIPT_DIR / 'outputs' / 'ml_model' / 'model_metadata.json'
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        validation = metadata.get('validation_metrics') or {}
        result = {
            "success": True,
            "available": True,
            "stock_code": stock_code,
            "year": year_int,
            "model": reference.get('model') or metadata.get('model'),
            "ml_probability": reference.get('ml_probability'),
            "ml_prediction": reference.get('ml_prediction'),
            "classification_threshold": reference.get('classification_threshold'),
            "shap_values": reference.get('shap_values') or [],
            "reference_note": reference.get('reference_note', ''),
            "feature_count": len(metadata.get('feature_columns') or []),
            "validation_metrics": {
                "pr_auc": validation.get('validation_pr_auc'),
                "roc_auc": validation.get('validation_roc_auc'),
                "precision": validation.get('precision'),
                "recall": validation.get('recall'),
                "f_beta": validation.get('f_beta'),
            },
        }
        if len(_ml_reference_cache) >= 32:
            _ml_reference_cache.clear()
        _ml_reference_cache[cache_key] = result
        return jsonify(_json_safe(result))
    except FileNotFoundError:
        # 模型文件缺失不应阻断规则底稿和三智能体分析。
        return jsonify({
            "success": True,
            "available": False,
            "stock_code": stock_code,
            "year": year_int,
            "message": "暂未找到可用的冻结模型。",
        })
    except Exception:
        app.logger.exception("ML 单样本参考结果加载失败")
        return jsonify({
            "success": True,
            "available": False,
            "stock_code": stock_code,
            "year": year_int,
            "message": "机器学习参考结果暂不可用。",
        })


@app.route('/api/analyze', methods=['POST'])
def full_analysis():
    """完整三智能体分析（工作底稿生成 + 指控 + 辩护 + 裁决）"""
    data = request.get_json(silent=True) or {}
    stock_code = str(data.get('code', '')).strip()
    year = str(data.get('year', '')).strip()

    if not stock_code or not year:
        return jsonify({"error": "请输入股票代码和年份"}), 400

    try:
        year_int = int(year)
    except (TypeError, ValueError):
        return jsonify({"error": "年份格式无效"}), 400

    stock_code = stock_code.zfill(6)

    api_key = os.environ.get('DEEPSEEK_API_KEY', '')
    if not api_key:
        return jsonify({"error": "未配置 DEEPSEEK_API_KEY"}), 500

    try:
        import three_agents_three_level as ta

        # 1. 生成工作底稿
        workpaper = get_workpaper_generator().generate(stock_code, year_int)
        if not workpaper.get('success'):
            return jsonify({"error": workpaper.get('error', '底稿生成失败')}), 500

        # 2. 三智能体分析
        orchestrator = ta.ThreeAgentOrchestrator(api_key=api_key)
        result = orchestrator.analyze(workpaper, verbose=False)

        if not result.get('success'):
            return jsonify({"error": result.get('error', '分析失败')}), 500

        return jsonify(_json_safe(result))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"分析失败: {str(e)}"}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("  基于三智能体协同-博弈的财务舞弊风险预警系统")
    print("  访问 http://localhost:5001")
    print("=" * 50)
    port = int(os.environ.get('PORT', '5001'))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
