"""
srforge
=======
多轮符号回归公式的结构化分析与公式锻造工具。
"""

# 数据模型
from .models import Formula, FormulaNode, Pattern, AnalysisContext

# 输入构建
from .builder import build_formulas, save_equations, load_equations

# 公式解析
from .parser import parse_formulas

# 分析引擎
from .analyzer import run_analyzer, print_reports

# 报告导出
from .report import export_html

# 公式评分
from .scorer import score_formulas, print_top_formulas

# 整合
from .pipeline import full_analysis, quick_score, round2_forge

# 子式提取
from .subexpr import extract_candidates, deduplicate_scored, print_candidates, augment_features, expand_feats

# Elastic Net 筛选
from .selector import elastic_select, print_selection

# 稳定性验证
from .validate import cross_validate, print_cv, check_formula
