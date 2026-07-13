"""
公式评分模块
综合 loss、complexity 和结构质量，为每条公式打分排序。

使用方式:
    from srforge import score_formulas, print_top_formulas

    scored = score_formulas(formulas, nodes, patterns, context)
    print_top_formulas(scored)
"""

from collections import defaultdict
from sympy import Symbol, Number
from .models import Formula, FormulaNode, Pattern, AnalysisContext
from .analyzer import build_children_map


def _group_nodes_by_formula(nodes: list[FormulaNode]) -> dict[int, list[FormulaNode]]:
    """formula_id → 该公式的所有节点"""
    groups = defaultdict(list)
    for n in nodes:
        groups[n.formula_id].append(n)
    return dict(groups)


def _build_pattern_index(patterns: list[Pattern]) -> dict[str, Pattern]:
    """signature → Pattern 快速查找"""
    return {p.signature: p for p in patterns}


def _slot_match_score(node: FormulaNode,
                      pattern: Pattern,
                      children_map: dict[int, list[FormulaNode]]) -> float:
    """
    计算单个节点的 slot 匹配度 (0~1)

    对照 pattern.slot_stats，看节点每个子位置的实际变量
    是否命中 slot 统计中的主导分布。
    """
    if not pattern.slot_stats:
        return 0.0

    children = children_map.get(node.node_id, [])
    scores = []

    for idx, child in enumerate(children):
        slot_key = f"arg_{idx}"

        # 变量名：Symbol 直接用名，Number 统一映射为 <const>
        if isinstance(child.expr, Symbol):
            var_name = str(child.expr)
        elif isinstance(child.expr, Number):
            var_name = "<const>"
        else:
            continue

        slot_dist = pattern.slot_stats.get(slot_key, {})
        if not slot_dist:
            continue

        total = sum(slot_dist.values())
        if total == 0:
            continue

        # slot 分布中该变量的占比 = 匹配度
        match_ratio = slot_dist.get(var_name, 0) / total
        scores.append(match_ratio)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def score_formulas(
    formulas: list[Formula],
    nodes: list[FormulaNode],
    patterns: list[Pattern],
    context: AnalysisContext,
    *,
    w_loss: float = 0.6,
    w_complexity: float = 0.1,
    w_structure: float = 0.3,
) -> list[dict]:
    """
    为每条公式计算综合评分（降序排列）

    formulas  : 公式列表
    nodes     : 全部 FormulaNode
    patterns  : build_patterns 产出（建议传过滤前）
    context   : build_context 产出

    返回 list[dict]，每项包含：
        formula_id, run_id, equation, loss, complexity,
        loss_score, complexity_score, structure_score, final_score
    """

    # 准备工作
    node_groups = _group_nodes_by_formula(nodes)
    pattern_index = _build_pattern_index(patterns)
    children_map = build_children_map(nodes)

    # 归一化参考值
    all_losses = [f.loss for f in formulas if f.loss is not None]
    all_complexities = [f.complexity for f in formulas if f.complexity is not None]

    max_loss = max(all_losses) if all_losses else 1.0
    min_loss = min(all_losses) if all_losses else 0.0
    max_complexity = max(all_complexities) if all_complexities else 1
    min_complexity = min(all_complexities) if all_complexities else 0

    results = []

    for formula in formulas:

        # --- 结构分：公式所有非叶节点的 slot 匹配度 ---
        formula_nodes = node_groups.get(formula.formula_id, [])
        structure_scores = []

        for node in formula_nodes:
            if node.is_leaf:
                continue

            sig = node.signature
            pattern = pattern_index.get(sig)
            if pattern is None:
                continue

            slot_match = _slot_match_score(node, pattern, children_map)
            structure_scores.append(slot_match)

        structure_score = (
            sum(structure_scores) / len(structure_scores)
            if structure_scores else 0.0
        )

        # --- loss 归一化（越低越好 → 得分越高） ---
        if max_loss != min_loss:
            loss_norm = 1 - (formula.loss - min_loss) / (max_loss - min_loss)
        else:
            loss_norm = 1.0

        # --- complexity 归一化（越低越好 → 得分越高） ---
        if max_complexity != min_complexity:
            comp_norm = 1 - (formula.complexity - min_complexity) / (max_complexity - min_complexity)
        else:
            comp_norm = 1.0

        # --- 最终分 ---
        final = (
            w_loss * loss_norm
            + w_complexity * comp_norm
            + w_structure * structure_score
        )

        results.append({
            "formula_id": formula.formula_id,
            "run_id": formula.run_id,
            "equation": formula.equation,
            "loss": formula.loss,
            "complexity": formula.complexity,
            "loss_score": round(loss_norm, 4),
            "complexity_score": round(comp_norm, 4),
            "structure_score": round(structure_score, 4),
            "final_score": round(final, 4),
        })

    # 降序排列
    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results


def print_top_formulas(scored: list[dict], topk: int = 10):
    """
    打印 top-k 公式（终端快速查看）
    """
    print(f"\n{'=' * 90}")
    print(f"{'Rank':<6} {'Formula':<42} {'Loss':<12} {'Comp':<8} {'Struct':<8} {'Final':<8}")
    print(f"{'=' * 90}")

    for i, r in enumerate(scored[:topk], 1):
        eq = r["equation"]
        if len(eq) > 40:
            eq = eq[:37] + "..."
        print(
            f"{i:<6} {eq:<42} "
            f"{r['loss']:<12.6f} {r['complexity']:<8} "
            f"{r['structure_score']:<8.3f} {r['final_score']:<8.3f}"
        )
