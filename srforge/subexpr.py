"""
子式提取与候选池构建
从 scored / patterns 中提取稳定子式，去重后输出候选池，
供 Elastic Net 或 Optuna 进一步筛选组装。
"""

import re
from collections import defaultdict, Counter
from sympy import Symbol, Number

from .analyzer import _build_signature, score_pattern
from .models import Formula, Pattern, AnalysisContext


def _to_structure(eq_str: str) -> str:
    """将公式字符串中的常数全部替换为 <c>，用于结构去重"""
    return re.sub(r"\d+\.?\d*", "<c>", eq_str)


def deduplicate_scored(scored: list[dict]) -> list[dict]:
    """
    结构去重：同一结构签名只保留 final_score 最高的一条。
    """
    best = {}
    for item in scored:
        struct = _to_structure(item["equation"])
        if struct not in best or item["final_score"] > best[struct]["final_score"]:
            best[struct] = item

    # 保持原有排序（按 final_score 降序）
    return sorted(best.values(), key=lambda x: x["final_score"], reverse=True)


def _has_extreme_pow(expr, max_exp: float = 4.0) -> bool:
    """检查表达式树中是否有 Pow 节点的常指数 |c| > max_exp"""
    from sympy import Pow, Number
    if isinstance(expr, Pow):
        base, exp = expr.args
        if isinstance(exp, Number) and abs(float(exp)) > max_exp:
            return True
    for child in expr.args:
        if _has_extreme_pow(child, max_exp):
            return True
    return False


def _extract_subexprs(expr, min_vars: int = 2, max_pow_exp: float = 4.0):
    """
    递归提取表达式树中所有非叶子子式。

    expr       : SymPy 表达式
    min_vars   : 最少包含几个特征变量，过滤过于简单的子式
    max_pow_exp: 允许的 Pow 常指数最大绝对值，超过的被跳过

    返回 [(子式expr, 结构签名, 特征名集合), ...]
    """
    results = []

    if isinstance(expr, (Symbol, Number)):
        return results

    free = {str(s) for s in expr.free_symbols}

    if len(free) >= min_vars and not _has_extreme_pow(expr, max_pow_exp):
        sig = _build_signature(expr)
        results.append((expr, sig, free))

    for child in expr.args:
        results.extend(_extract_subexprs(child, min_vars, max_pow_exp))

    return results


def extract_candidates(
    formulas: list[Formula],
    scored: list[dict],
    patterns: list[Pattern],
    context: AnalysisContext,
    *,
    min_vars: int = 2,
    top_n_scored: int = 30,
    max_pow_exp: float = 4.0,
    x_filter=None,
) -> list[dict]:
    """
    从 scored 公式中提取候选子式池。

    formulas : 原始 Formula 对象列表（含 SymPy expr）
    scored   : score_formulas() 的输出
    patterns : build_patterns() 的输出（建议未过滤）
    context  : build_context() 的输出
    min_vars : 子式至少包含几个特征
    top_n_scored : 取 scored 前多少条公式来扒子式（去重前）
    max_pow_exp  : Pow 常指数最大绝对值，超过会被过滤
    x_filter     : 训练集 DataFrame，用于过滤会产生 NaN/Inf 的子式

    返回 list[dict]，每项：
        signature     : 结构签名
        subexpr       : 子式字符串（取最高频的常数值版本）
        features      : 包含的特征名列表
        pattern_score : 对应 pattern 的评分
        frequency     : 子式出现次数
        n_features    : 特征数
    """

    # 1. 准备查找表
    formula_map = {f.formula_id: f for f in formulas}
    pattern_map = {p.signature: p for p in patterns}

    # 2. 对 scored 做结构去重
    deduped = deduplicate_scored(scored[:top_n_scored])

    # 3. 遍历去重后的公式，提取子式
    #    sig → [{"subexpr": str, "features": [...]}]
    pool = defaultdict(list)

    for item in deduped:
        fid = item["formula_id"]
        formula = formula_map.get(fid)
        if formula is None:
            continue

        for expr, sig, free in _extract_subexprs(formula.expr, min_vars, max_pow_exp):
            pool[sig].append({
                "subexpr": str(expr),
                "features": sorted(free),
                "formula_score": item["final_score"],
            })

    # 4. 汇总：每类结构取最高频的子式字符串，
    #    并查 pattern 评分和频次
    candidates = []

    for sig, items in pool.items():
        # 频次最高的具体子式（含常数）
        eq_counter = Counter(item["subexpr"] for item in items)
        best_eq, occurrences = eq_counter.most_common(1)[0]
        best_features = items[0]["features"]

        # 查 pattern
        pat = pattern_map.get(sig)
        if pat is not None:
            pat_score = score_pattern(pat, context)
            pat_freq = pat.frequency
        else:
            pat_score = 0.0
            pat_freq = 0

        candidates.append({
            "signature": sig,
            "subexpr": best_eq,
            "features": best_features,
            "pattern_score": round(pat_score, 4),
            "frequency": pat_freq,
            "occurrences": occurrences,
            "n_features": len(best_features),
        })

    # 5. 按 pattern_score + frequency 综合排序
    candidates.sort(
        key=lambda c: (c["pattern_score"], c["frequency"], c["occurrences"]),
        reverse=True,
    )

    # 6. 过滤会产生 NaN/Inf 的子式（如分母接近零的除法）
    if x_filter is not None:
        import numpy as np
        xf = x_filter.values if hasattr(x_filter, "values") else np.asarray(x_filter)
        cols = x_filter.columns.tolist() if hasattr(x_filter, "columns") else []
        col_idx = {c: i for i, c in enumerate(cols)}

        stable = []
        for c in candidates:
            namespace = {feat: xf[:, col_idx[feat]] for feat in c["features"]}
            namespace.update({
                "sqrt": np.sqrt, "log": np.log, "exp": np.exp,
                "sin": np.sin, "cos": np.cos, "abs": np.abs,
                "square": np.square,
            })
            try:
                vals = eval(c["subexpr"], {"__builtins__": {}}, namespace)
                vals = np.asarray(vals, dtype=float).ravel()
                if np.all(np.isfinite(vals)):
                    stable.append(c)
            except Exception:
                pass

        n_dropped = len(candidates) - len(stable)
        if n_dropped:
            print(f"[subexpr] 过滤掉 {n_dropped} 个不稳定的子式（NaN/Inf）")
        candidates = stable

    return candidates


def augment_features(candidates: list[dict], x_train, x_test, top_n: int = 8):
    """
    将 top-N 候选子式作为新特征列，加到训练/测试数据中。
    用于 PySR 第二轮迭代。

    返回 (x_train_new, x_test_new, column_map)
        column_map: {"feat_1": "GGBS + OPC", "feat_2": "sqrt(...)", ...}
    """
    import numpy as np
    import pandas as pd

    x_tr = x_train.copy()
    x_te = x_test.copy()
    col_names = x_train.columns.tolist() if hasattr(x_train, "columns") else []

    col_map = {}
    used = set()

    idx = 1
    for c in candidates[:top_n]:
        se = c["subexpr"]

        # 跳过和已有列同名的子式
        if se in col_names or se in used:
            continue
        used.add(se)

        feat_name = f"feat_{idx}"
        idx += 1
        col_map[feat_name] = se

        for df in [x_tr, x_te]:
            namespace = {col: df[col].values for col in df.columns}
            namespace.update({
                "sqrt": np.sqrt, "log": np.log, "exp": np.exp,
                "sin": np.sin, "cos": np.cos, "abs": np.abs,
                "square": np.square,
            })
            try:
                vals = eval(se, {"__builtins__": {}}, namespace)
                vals = np.asarray(vals, dtype=float).ravel()
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                vals = np.zeros(len(df))
            df[feat_name] = vals

    return x_tr, x_te, col_map


def expand_feats(equation: str, feat_map: dict) -> str:
    """
    将公式中的 feat_1, feat_2 等还原为原始子式表达式。

    equation : 含 feat_N 的公式字符串
    feat_map : augment_features 返回的 column_map

    返回可直接 eval 的公式字符串。
    """
    eq = equation
    for name, expr in feat_map.items():
        eq = eq.replace(name, f"({expr})")
    return eq


def print_candidates(candidates: list[dict], topk: int = 15):
    """终端打印候选子式"""
    print(f"\n{'='*80}")
    print(f"{'#':<4} {'Subexpr':<45} {'Pat':<7} {'Freq':<6} {'nFeat':<6}")
    print(f"{'='*80}")
    for i, c in enumerate(candidates[:topk], 1):
        se = c["subexpr"]
        if len(se) > 43:
            se = se[:40] + "..."
        print(
            f"{i:<4} {se:<45} "
            f"{c['pattern_score']:<7.3f} {c['frequency']:<6} {c['n_features']:<6}"
        )
