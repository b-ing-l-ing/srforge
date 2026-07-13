"""
跨 seed 稳定性验证
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def cross_validate(equation: str, X, y, seeds=range(0, 51, 5),
                   threshold=3.0, stratify=True) -> pd.DataFrame:
    """
    对一条公式跨 seed 跑分，返回每轮 R²/MSE/MAE 和测试集分布。

    equation  : 公式字符串
    X, y      : 完整数据集
    seeds     : 测试的 random_state 列表
    threshold : IQR 异常值截断倍数
    stratify  : 是否分层（确保每轮 y 分布一致）

    返回 DataFrame，列: seed, r2, mse, mae, y_mean, y_std, n_bad
    """
    funcs = {
        "sqrt": np.sqrt, "log": np.log, "exp": np.exp,
        "sin": np.sin, "cos": np.cos, "abs": np.abs,
        "square": np.square,
    }

    y_all = y.values.ravel() if hasattr(y, "values") else np.asarray(y).ravel()
    X_all = X.values if hasattr(X, "values") else np.asarray(X)
    cols = X.columns.tolist() if hasattr(X, "columns") else [f"x{j}" for j in range(X.shape[1])]

    rows = []

    for seed in seeds:
        x_tr, x_te, y_tr, y_te, idx_tr, idx_te = _split_with_idx(X_all, y_all, seed, stratify)

        # IQR clip
        for j in range(x_tr.shape[1]):
            q1, q3 = np.quantile(x_tr[:, j], [0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
            x_te[:, j] = np.clip(x_te[:, j], lo, hi)

        ns = {cols[i]: x_te[:, i] for i in range(len(cols))}
        ns.update(funcs)

        yp = eval(equation, {"__builtins__": {}}, ns)
        yp = np.asarray(yp, dtype=float).ravel()
        ok = np.isfinite(yp)

        if ok.sum() == 0:
            rows.append({"seed": seed, "r2": np.nan, "mse": np.nan, "mae": np.nan,
                         "y_mean": np.nan, "y_std": np.nan, "n_bad": len(y_te)})
            continue

        mse = mean_squared_error(y_te[ok], yp[ok])
        r2 = r2_score(y_te[ok], yp[ok])
        mae = mean_absolute_error(y_te[ok], yp[ok])
        bad = (np.abs(y_te[ok] - yp[ok]) > 2 * np.abs(y_te[ok] - yp[ok]).std()).sum()

        rows.append({
            "seed": seed,
            "r2": round(r2, 4), "mse": round(mse, 2), "mae": round(mae, 2),
            "y_mean": round(y_te.mean(), 2), "y_std": round(y_te.std(), 2),
            "n_bad": int(bad),
        })

    return pd.DataFrame(rows)


def _split_with_idx(X, y, seed, stratify):
    n = len(X)
    idx = np.arange(n)
    if stratify and len(np.unique(y)) >= 5:
        y_bins = pd.qcut(y, q=5, labels=False)
    else:
        y_bins = None
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.3, random_state=seed,
        stratify=y_bins,
    )
    return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx], tr_idx, te_idx


def check_formula(equation: str, X) -> list[str]:
    """
    快速检查公式安全性。返回空列表 = 安全。

    检查项：
    1. Pow 指数含变量（如 Na_Al**K2O）
    2. 分母中 X-const 是否接近零
    """
    from sympy import sympify, Pow, Symbol

    warnings = []
    cols = X.columns.tolist() if hasattr(X, "columns") else []
    vals_map = {c: X[c].values for c in cols}

    # 1. 变量指数
    try:
        expr = sympify(equation)
        for node in _walk_nodes(expr):
            if isinstance(node, Pow):
                _, exp_node = node.args
                if isinstance(exp_node, Symbol):
                    warnings.append(f"变量指数: {node}")
    except Exception:
        pass

    # 2. 分母接近零
    for col in cols:
        vals = vals_map[col]
        # 匹配 patterns like "col - 0.004" or "0.004 - col" after /
        import re
        for pattern in [rf'(?<=[/\s(]){col}\s*-\s*([\d.]+)', rf'(?<=[/\s(])([\d.]+)\s*-\s*{col}']:
            for m in re.finditer(pattern, equation):
                const = float(m.group(1))
                dist = np.abs(vals - const).min()
                if dist < 0.01:
                    warnings.append(
                        f"{col} 接近 {const:.4f}，min_dist={dist:.4f}"
                    )

    return warnings


def _walk_nodes(expr):
    """递归遍历 SymPy 节点"""
    yield expr
    for arg in expr.args:
        yield from _walk_nodes(arg)


def print_cv(df: pd.DataFrame):
    """打印 cross_validate 结果"""
    print(f"{'seed':<8} {'R²':<9} {'MSE':<10} {'y_mean':<10} {'y_std':<10} {'bad':<6}")
    print("-" * 55)
    for _, r in df.iterrows():
        flag = " ← LOW" if r["r2"] < 0.4 else (" ★" if r["r2"] > 0.7 else "")
        print(f"{r['seed']:<8} {r['r2']:<9.4f} {r['mse']:<10.2f} {r['y_mean']:<10.2f} {r['y_std']:<10.2f} {r['n_bad']:<6}{flag}")
    print(f"\n均值 R²={df['r2'].mean():.4f}  最佳={df['r2'].max():.4f}  最差={df['r2'].min():.4f}")
