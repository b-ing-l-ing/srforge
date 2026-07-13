"""
整合模块
一键运行完整分析流程，避免手动拼接各步骤。
"""

from pathlib import Path

from .parser import parse_formulas
from .analyzer import (
    annotate_signatures,
    build_patterns,
    build_context,
    run_analyzer,
    print_reports,
)
from .scorer import score_formulas, print_top_formulas
from .report import export_html
from .subexpr import extract_candidates, augment_features
from .builder import build_formulas, save_equations


def full_analysis(
    formulas,
    *,
    output_html: str | Path | None = "report.html",
    open_browser: bool = True,
    topk: int = 20,
    verbose: bool = True,
) -> dict:
    """
    完整分析流程：解析 → Pattern分析 → 公式评分 → HTML报告

    formulas    : Formula 对象列表（由 build_formulas 或手写）
    output_html : HTML 报告路径，传 None 则跳过导出
    open_browser: 是否自动打开浏览器
    topk        : 终端打印 top-k
    verbose     : 是否在终端打印 Pattern 报告和公式排名

    返回 {"reports": ..., "scored": ..., "context": ...}
    """

    # 1. 解析
    nodes = parse_formulas(formulas)

    # 2. 签名 + 未过滤 patterns（给 scorer 用）
    annotate_signatures(nodes)
    patterns = build_patterns(nodes)
    context = build_context(patterns)

    # 3. 公式评分
    scored = score_formulas(formulas, nodes, patterns, context)

    # 4. Pattern 报告（内部会过滤）
    reports = run_analyzer(nodes)

    # 5. 终端输出
    if verbose:
        print_reports(reports)
        print_top_formulas(scored, topk=topk)

    # 6. HTML 导出
    if output_html is not None:
        export_html(reports, output_path=output_html,
                    open_browser=open_browser, scored=scored)

    return {"reports": reports, "scored": scored, "context": context, "patterns": patterns}


def quick_score(formulas, *, topk: int = 20) -> list[dict]:
    """
    只做公式评分，不生成报告。
    适合快速对比不同参数下的公式质量。
    """

    nodes = parse_formulas(formulas)
    annotate_signatures(nodes)
    patterns = build_patterns(nodes)
    context = build_context(patterns)

    scored = score_formulas(formulas, nodes, patterns, context)
    print_top_formulas(scored, topk=topk)
    return scored


def round2_forge(
    result: dict,
    formulas,
    x_train,
    y_train,
    x_test,
    y_test,
    train_fn,
    *,
    n_rounds: int = 30,
    top_n: int = 8,
    save_path: str = "equations_round2.pkl",
    output_html: str | None = "report_round2.html",
) -> dict:
    """
    提取共识子式 → 造新特征 → PySR Round 2 → 分析。

    result   : full_analysis 的返回值
    formulas : 原始 Formula 对象列表
    x_train, y_train, x_test, y_test : 训练/测试数据
    train_fn : def train(x_train, y_train, n) -> list_of_equations_dfs
    n_rounds : Round 2 PySR 轮数
    top_n    : 取前 N 个候选子式作为新特征

    返回 {"feat_map": ..., "x_train": ..., "x_test": ...,
          "reports": ..., "scored": ..., "patterns": ..., "context": ...}
    """

    candidates = extract_candidates(
        formulas, result["scored"], result["patterns"], result["context"],
        x_filter=x_train,
    )

    x_tr2, x_te2, feat_map = augment_features(candidates, x_train, x_test,
                                               top_n=top_n)

    list2 = train_fn(x_tr2, y_train, n_rounds)
    save_equations(list2, filepath=save_path)
    formulas2 = build_formulas(list2)
    result2 = full_analysis(formulas2, output_html=output_html, verbose=False)

    return {
        "feat_map": feat_map,
        "formulas": formulas2,
        "x_train": x_tr2,
        "x_test": x_te2,
        **result2,
    }
