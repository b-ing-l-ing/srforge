from sympy import Expr
from sympy import Symbol
from sympy import Number
from collections import defaultdict, Counter
import math
import statistics

from .models import FormulaNode, Pattern, AnalysisContext


# todo 数据生成
def _build_signature(expr:Expr) -> str:
    """
    递归构建SymPy表达式的结构标签
    """
    if isinstance(expr, Symbol):
        return "VAR"

    if isinstance(expr,Number):
        return "CONST"

    # 递归
    children = [_build_signature(child) for child in expr.args]

    # 叶节点
    if len(children) == 0:
        return expr.func.__name__

    # 非叶节点
    return f"{expr.func.__name__}({','.join(children)})"

def build_signature(node: FormulaNode) -> str:
    """
    根据 FormulaNode 构建 结构标签
    """
    return _build_signature(node.expr)


def annotate_signatures(nodes: list[FormulaNode]) -> None:
    """
    为每个 FormulaNode 添加 Signature。
    """

    for node in nodes:
        node.signature = build_signature(node)


def build_children_map(nodes: list[FormulaNode]):
    children = defaultdict(list)
    for n in nodes:
        if n.parent_id is not None:
            children[n.parent_id].append(n)
    return children

def analyze_slots(all_nodes: list[FormulaNode],
                  group_nodes: list[FormulaNode]):
    """
    slot = parent signature + 子节点相对位置
    value = Symbol

    all_nodes : 全部节点，用于构建完整的 children_map
    group_nodes : 当前 signature 组的节点，用于定位 slot
    """

    children_map = build_children_map(all_nodes)

    table = defaultdict(lambda: defaultdict(Counter))

    for node in group_nodes:

        if node.is_leaf:
            continue

        sig = node.signature

        children = children_map.get(node.node_id, [])

        for idx, child in enumerate(children):

            if isinstance(child.expr, Symbol):
                var_name = str(child.expr)
            elif isinstance(child.expr, Number):
                var_name = "<const>"
            else:
                continue

            slot_key = f"arg_{idx}"
            table[sig][slot_key][var_name] += 1

    return table

def classify_signature(sig: str) -> str:
    """
    给 signature 分类结构类型
    """

    # 1. 基础叶子
    if sig == "VAR":
        return "VAR_ONLY"
    if sig == "CONST":
        return "CONST_ONLY"

    # 2. 一元函数
    if sig.startswith(("sin(", "cos(", "tan(",
                       "exp(", "log(", "sqrt(",
                       "abs(", "sign(",
                       "sinh(", "cosh(", "tanh(",
                       "asin(", "acos(", "atan(")):
        return "UNARY"

    # 3. 幂结构
    if sig.startswith("Pow"):
        return "POWER"

    # 4. 加法结构
    if sig.startswith("Add"):
        if "Mul" in sig:
            return "COMPOSITE"
        return "LINEAR"

    # 5. 乘法结构
    if sig.startswith("Mul"):
        return "MULTIPLICATIVE"

    return "OTHER"

def build_patterns(nodes: list[FormulaNode]) -> list[Pattern]:
    """
    从 FormulaNode 列表构建 Pattern 列表
    """

    # signature -> nodes
    grouped = defaultdict(list)

    for node in nodes:
        grouped[node.signature].append(node)

    patterns = []

    for sig, group_nodes in grouped.items():

        # frequency
        freq = len(group_nodes)

        # formula_ids
        formula_ids = sorted({
            node.formula_id for node in group_nodes
        })

        # run_ids
        run_ids = sorted({
            node.run_id for node in group_nodes
        })

        # slot统计 
        slot_table = analyze_slots(nodes, group_nodes)

        slot_stats = {
            slot: dict(counter)
            for slot, counter in slot_table[sig].items()
        }

        for slot,dist in slot_stats.items():
            for k, v in dist.items():
                if not isinstance(v, int):
                    raise TypeError(
                        f"slot_stats corrupted: {slot} {k} {type(v)}"
                    )

        patterns.append(
            Pattern(
                signature=sig,
                frequency=freq,
                slot_stats=slot_stats,
                type=classify_signature(sig),
                formula_ids = formula_ids,
                run_ids = run_ids
            )
        )

    return patterns

def build_context(patterns: list[Pattern]) -> AnalysisContext:
    """
    根据 Pattern 列表构建全局分析上下文
    """
    all_run_ids = set()
    all_formula_ids = set()

    max_frequency = 0

    for pattern in patterns:
        all_run_ids.update(pattern.run_ids)
        all_formula_ids.update(pattern.formula_ids)

        max_frequency = max(max_frequency, pattern.frequency)

    return AnalysisContext(
        total_runs = len(all_run_ids),
        total_formulas = len(all_formula_ids),
        max_frequency = max_frequency
    )



# -----------------------------------------------------
# todo pattern评价

def compute_slot_stability(slot_stats: dict) -> float:
    """
    计算 slot 的稳定性(0~1)
    """

    if not slot_stats:
        return 0.0

    scores = []

    for slot,counter in slot_stats.items():

        total = sum(counter.values())
        if total == 0:
            continue

        dominant = max(counter.values())

        ratio = dominant / total    # 该变量的主导性
        scores.append(ratio)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def compute_run_score(pattern: Pattern, context: AnalysisContext) -> float:
    """
    Run Coverage（0~1）
    """
    if context.total_runs == 0:
        return 0.0
    return len(pattern.run_ids) / context.total_runs


def compute_formula_score(pattern: Pattern, context: AnalysisContext) -> float:
    """
    Formula Coverage（0~1）
    """
    if context.total_formulas == 0:
        return 0.0
    return len(pattern.formula_ids) / context.total_formulas


def compute_frequency_score(pattern: Pattern, context: AnalysisContext) -> float:
    """
    Frequency（0~1）
    """
    if context.max_frequency == 0:
        return 0.0
    return pattern.frequency / context.max_frequency


def score_pattern_detail(
    pattern: Pattern,
    context: AnalysisContext,
) -> dict:
    """
    返回 Pattern 各项评分，用于 Report 展示。
    """

    run_score = compute_run_score(pattern, context)
    formula_score = compute_formula_score(pattern, context)
    stability_score = compute_slot_stability(pattern.slot_stats)
    frequency_score = compute_frequency_score(pattern, context)
    pattern_conf = compute_pattern_confidence(pattern, context)

    RUN_WEIGHT = 0.4
    FORMULA_WEIGHT = 0.3
    STABILITY_WEIGHT = 0.2
    FREQUENCY_WEIGHT = 0.1

    total_score = (
        RUN_WEIGHT * run_score
        + FORMULA_WEIGHT * formula_score
        + STABILITY_WEIGHT * stability_score
        + FREQUENCY_WEIGHT * frequency_score
    )

    return {
        "total": total_score,
        "run": run_score,
        "formula": formula_score,
        "stability": stability_score,
        "frequency": frequency_score,
        "pattern_confidence": pattern_conf
    }


def score_pattern(pattern: Pattern, context: AnalysisContext) -> float:
    """
    返回 Pattern 综合评分
    """
    return score_pattern_detail(pattern, context)["total"]


def rank_patterns(patterns: list[Pattern], context: AnalysisContext) -> list[Pattern]:
    """
    对 Pattern 列表进行排序
    """
    return sorted(
        patterns,
        key=lambda p: score_pattern(p, context),
        reverse=True
    )


def interpret_pattern(pattern: Pattern, context: AnalysisContext) -> dict:
    """
    将 Pattern 转换为“可解释报告”
    """

    report = {}

    report["signature"] = pattern.signature
    report["type"] = pattern.type
    report["frequency"] = pattern.frequency

    report["formula_ids"] = pattern.formula_ids
    report["run_ids"] = pattern.run_ids

    report["formula_count"] = len(pattern.formula_ids)
    report["run_count"] = len(pattern.run_ids)

    report["score"] = score_pattern_detail(pattern,context)
    report["coverage"] = {
    "run": {
        "count": report["run_count"],
        "total": context.total_runs,
        "ratio": report["run_count"] / context.total_runs,
    },
    "formula": {
        "count": report["formula_count"],
        "total": context.total_formulas,
        "ratio": report["formula_count"] / context.total_formulas,
    },
    }

    # slot解释
    slot_summary = build_slot_summary(pattern)
    report["slots"] = slot_summary
    report["slot_manual_view"] = build_manual_slot_view(pattern)

    # 生成自然语言解释

    report["interpretation"] = generate_interpretation(report)

    return report

def generate_interpretation(report):

    text = []
    text.append(f"综合评分：\t{report['score']['total']:.3f}")
    text.append(f"结构模式：\t{report['signature']}")
    text.append(f"出现频率：\t{report['frequency']}")
    text.append(f"覆盖 Run 数: \t{report['run_count']}")

    # 为稳定性评价
    if report["score"]["stability"] > 0.8:
        text.append("结构非常稳定，变量分布高度一致")
    elif report["score"]["stability"] > 0.5:
        text.append("结构中等稳定，存在一定变量波动")
    else:
        text.append("结构较不稳定，变量分布较分散")


    for slot, info in report["slots"].items():
        text.append(
            f"{slot}主要变量是 {info['dominant']} "
            f"(占比 {info['ratio']})"
        )

    # 综合评价
    text.append("")

    if report["score"]["total"] >= 0.8:
        text.append("综合建议：优先作为候选结构进行分析。")

    elif report["score"]["total"] >= 0.6:
        text.append("综合建议：建议保留，后续结合 Slot 分析。")

    else:
        text.append("综合建议：暂不作为主要候选结构，可作为补充参考。")

    # 证据评估
    run_ratio = report["coverage"]["run"]["ratio"]
    formula_ratio = report["coverage"]["formula"]["ratio"]

    if run_ratio >= 0.8 and formula_ratio >= 0.2:
        text.append("")
        text.append("证据充分：")
        text.append(
            "该 Pattern 在多数 PySR Run 中重复出现，"
            "并覆盖了较多公式，可信度较高。"
        )

    elif run_ratio >= 0.5 or formula_ratio >= 0.1:
        text.append("")
        text.append("证据一般：")
        text.append(
            "该 Pattern 具有一定重复性，"
            "建议结合其它 Pattern 一起分析。"
        )

    else:
        text.append("")
        text.append("证据较弱：")
        text.append(
            "该 Pattern 仅在少量 Run 或公式中出现，"
            "目前建议作为参考结构。"
        )

    return "\n".join(text)

# -----------------------------------------------------
# todo 数据选择
def compute_entropy(counter: dict) -> float:
    """
    Shannon entropy(未归一化)
    """
    total = sum(counter.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for v in counter.values():
        p = v / total
        entropy -= p * math.log(p + 1e-9)

    return entropy

def compute_normalized_entropy(counter: dict) -> float:
    """
    归一化 entropy ∈ [0,1]
    用于不同 slot 可比性
    """
    total = sum(counter.values())
    if total == 0:
        return 0.0

    probs = [v / total for v in counter.values()]
    entropy = -sum(p * math.log(p + 1e-9) for p in probs)
    max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0

    return entropy / (max_entropy + 1e-9)


def build_slot_summary(pattern):
    """
    Slot summary + confidence
    """

    slot_summary = {}

    for slot, counter in pattern.slot_stats.items():

        if not counter:
            continue

        total = sum(counter.values())

        # 排序
        ranking = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        dominant_var, dominant_count = ranking[0]

        top1_ratio = dominant_count / total

        if len(ranking) >= 2:
            top2_ratio = ranking[1][1] / total
        else:
            top2_ratio = 0.0

        entropy = compute_entropy(counter)
        norm_entropy = compute_normalized_entropy(counter)
        confidence = compute_slot_confidence(counter)

        slot_summary[slot] = {
            "dominant": dominant_var,
            "ratio": round(top1_ratio, 3),
            "distribution": dict(counter),
            "ranking": ranking,
            "entropy": round(entropy, 4),
            "norm_entropy": round(norm_entropy, 4),
            "confidence": round(confidence, 4),
            "top_gap": round(top1_ratio - top2_ratio, 3),
        }

    return slot_summary


def filter_patterns(
    patterns: list[Pattern],
    context: AnalysisContext,
    min_freq: int =2,
    min_run_ratio: float =0.2
):
    """
    Pattern 去噪过滤器
    """

    filtered = []

    for p in patterns:

        # 过滤叶节点 Pattern
        if p.type in ("VAR_ONLY","CONST_ONLY"):
            continue

        # 出现次数过低
        if p.frequency < min_freq:
            continue

        # run 覆盖率低
        run_ratio = compute_run_score(p, context)

        if run_ratio < min_run_ratio:
            continue

        filtered.append(p)

    return filtered

def compute_slot_confidence(counter: dict) -> float:
    """
    Slot 可信度(0~1)

    组合三种信息：
    - top1 dominance
    - top1 vs top2 gap
    - entropy disorder
    """

    if not counter:
        return 0.0

    total = sum(counter.values())
    probs = [v / total for v in counter.values()]
    probs.sort(reverse=True)

    top1 = probs[0]
    top2 = probs[1] if len(probs) > 1 else 0.0

    norm_entropy = compute_normalized_entropy(counter)

    confidence = (
        0.5 * top1 +
        0.3 * (top1 - top2) +
        0.2 * (1 - norm_entropy)
    )

    return float(confidence)

def compute_pattern_confidence(pattern, context):
    """
    Pattern 级别可信度
    """

    slot_confidences = []

    for slot, counter in pattern.slot_stats.items():
        if not counter:
            continue

        c = compute_slot_confidence(counter)
        slot_confidences.append(c)

    if not slot_confidences:
        return 0.0

    # 1. 平均 slot confidence
    mean_conf = sum(slot_confidences) / len(slot_confidences)

    # 2. slot consistency
    if len(slot_confidences) > 1:
        std_conf = statistics.pstdev(slot_confidences)
        consistency = 1 - std_conf
    else:
        consistency = 1.0

    # 3. coverage
    run_score = compute_run_score(pattern, context)
    formula_score = compute_formula_score(pattern, context)
    coverage = (run_score + formula_score) / 2

    # 4. final pattern confidence
    pattern_confidence = (
        0.5 * mean_conf +
        0.3 * consistency +
        0.2 * coverage
    )

    return float(pattern_confidence)


def build_manual_slot_view(pattern):
    """
    人工分析专用视图（不影响任何评分逻辑）
    """

    view = {}

    for slot, counter in pattern.slot_stats.items():

        if not counter:
            continue

        total = sum(counter.values())

        ranking = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        top1_var, top1_count = ranking[0]
        top1_ratio = top1_count / total

        if len(ranking) > 1:
            top2_ratio = ranking[1][1] / total
        else:
            top2_ratio = 0.0

        view[slot] = {
            "top1": top1_var,
            "top1_ratio": round(top1_ratio, 3),
            "top2_ratio": round(top2_ratio, 3),
            "confidence": round(compute_slot_confidence(counter), 3),
            "distribution": dict(counter),
            "ranking": ranking
        }

    return view

# -----------------------------------------------------

# -----------------------------------------------------
def print_reports(reports, verbose=False):
    """
    打印 Pattern Report
    verbose=False : 简洁模式
    verbose=True  : 额外输出自然语言解释
    """

    for i, report in enumerate(reports, start=1):

        print("\n" + "=" * 70)
        print(f"Pattern #{i}")
        print("=" * 70)

        # ---------------- Basic ----------------

        print(f"Signature : {report['signature']}")
        print(f"Frequency : {report['frequency']}")

        # ---------------- Coverage ----------------

        run = report["coverage"]["run"]
        formula = report["coverage"]["formula"]

        print("\nCoverage")
        print(
            f"  Run     : {run['count']} / {run['total']} "
            f"({run['ratio']:.1%})"
        )
        print(
            f"  Formula : {formula['count']} / {formula['total']} "
            f"({formula['ratio']:.1%})"
        )

        # ---------------- Score ----------------

        score = report["score"]
        total_score = score["total"]

        print("\nScore")
        print(f"  Total      : {total_score:.3f}   综合评分")
        print(f"  Run        : {score['run']:.3f}   越接近1说明跨Run重复出现")
        print(f"  Formula    : {score['formula']:.3f}   越接近1说明覆盖更多公式")
        print(f"  Stability  : {score['stability']:.3f}   越接近1说明变量位置越稳定")
        print(f"  Frequency  : {score['frequency']:.3f}   出现频率(归一化)")

        # ---------------- Evidence ----------------

        print("\nEvidence")

        if run["ratio"] >= 0.8 and formula["ratio"] >= 0.2:
            print("  Strong")
        elif run["ratio"] >= 0.5 or formula["ratio"] >= 0.1:
            print("  Moderate")
        else:
            print("  Weak")

        # ---------------- Slots ----------------

        print("\nSlots")

        slots_data = report.get("slot_manual_view")

        if slots_data is None:
            slots_data = report.get("slots", {})

        if slots_data:

            for slot, info in slots_data.items():
                print(f"\n  {slot}")

                # ---------------- Human view ----------------
                if "distribution" in info:

                    total = sum(info["distribution"].values())

                    print("   Distribution:")

                    for var, count in info["distribution"].items():
                        ratio = count / total if total > 0 else 0
                        print(f"     {var:<6} {ratio:.1%}")

                    # optional enriched fields
                    if "confidence" in info:
                        print(f"   Confidence: {info['confidence']:.3f}")

                    if "top1_ratio" in info:
                        print(f"   Top1: {info['top1']} ({info['top1_ratio']:.2f})")

                    if "top2_ratio" in info:
                        print(f"   Top2 ratio: {info['top2_ratio']:.2f}")

                    if "entropy" in info:
                        print(f"   Entropy: {info['entropy']:.4f}")

                # ---------------- legacy fallback ----------------
                elif "ranking" in info:

                    total = sum(info["distribution"].values())

                    for var, count in info["ranking"]:
                        ratio = count / total if total > 0 else 0
                        print(f"     {var:<6} {ratio:.1%}")

                else:
                    print("     (no detail available)")

        else:
            print("  None")

        # ---------------- Recommendation ----------------

        print("\nRecommendation")

        if total_score >= 0.8:
            print("  优先候选")
        elif total_score >= 0.6:
            print("  建议保留")
        else:
            print("  作为参考")

        # ---------------- Verbose ----------------

        if verbose:
            print("\nInterpretation")
            print(report["interpretation"])
# -----------------------------------------------------
# todo 总入口
def run_analyzer(nodes):
    """
    从 FormulaNode → Report 的完整流程
    """

    annotate_signatures(nodes)

    patterns = build_patterns(nodes)

    context = build_context(patterns)

    patterns = filter_patterns(patterns,context)

    patterns = rank_patterns(patterns,context)

    reports = [interpret_pattern(p,context) for p in patterns]

    return reports
