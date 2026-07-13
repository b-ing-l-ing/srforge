from .models import Formula
from sympy import sympify, SympifyError


def build_formulas(list_of_equations):
    """
    将 PySR 多次训练结果转换为 Formula 对象列表
    """

    formulas = []
    formula_id = 0

    for run_id, eq_table in enumerate(list_of_equations):

        # PySR 的 equations_ 通常是 DataFrame
        # 每一行是一条公式
        for _, row in eq_table.iterrows():

            # PySR字段
            expr_str = row["equation"]
            loss = row.get("loss", 0.0) or 0.0
            complexity = row.get("complexity", 0) or 0

            try:
                expr = sympify(expr_str)
            except SympifyError:
                # 跳过 PySR 偶发的畸形公式
                continue

            formula = Formula(
                formula_id=formula_id,
                run_id=run_id,
                expr=expr,
                loss=loss,
                complexity=complexity
            )

            formulas.append(formula)
            formula_id += 1

    return formulas


def save_equations(equations_dfs, filepath="equations/equations.pkl"):
    """保存 PySR equations_ DataFrame 列表到磁盘"""
    import pickle
    from pathlib import Path

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump({"n_runs": len(equations_dfs), "equations": equations_dfs}, f)


def load_equations(filepath="equations/equations.pkl"):
    """从磁盘加载 equations_ DataFrame 列表"""
    import pickle

    with open(filepath, "rb") as f:
        return pickle.load(f)["equations"]