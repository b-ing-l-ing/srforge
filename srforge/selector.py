"""
Elastic Net 子式筛选模块
从候选子式池中自动选择最优组合，输出最终公式。
"""

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def _eval_candidates(candidates: list[dict], X: np.ndarray, col_map: dict) -> np.ndarray:
    """
    将候选子式在数据集上求值，构建设计矩阵。

    返回 (n_samples, n_candidates) 的数组。
    col_map: 列名 → 列索引
    """
    n_samples = len(X)
    X_f = np.zeros((n_samples, len(candidates)))

    for j, c in enumerate(candidates):
        eq = c["subexpr"]
        # 只提供子式中出现的列，减少 eval 开销
        namespace = {feat: X[:, col_map[feat]] for feat in c["features"]}
        namespace.update({
            "sqrt": np.sqrt, "log": np.log, "exp": np.exp,
            "sin": np.sin, "cos": np.cos, "abs": np.abs,
            "square": np.square,
        })
        try:
            vals = eval(eq, {"__builtins__": {}}, namespace)
            vals = np.asarray(vals, dtype=float).ravel()
            if np.any(~np.isfinite(vals)):
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            X_f[:, j] = vals
        except Exception:
            X_f[:, j] = 0.0

    return X_f


def elastic_select(
    candidates: list[dict],
    x_train,
    y_train,
    x_test=None,
    y_test=None,
    *,
    l1_ratio: tuple = (0.1, 0.5, 0.7, 0.9, 0.95, 1.0),
    top_n: int = 30,
) -> dict:
    """
    Elastic Net 自动筛选子式，输出最终公式。

    candidates : extract_candidates 的输出
    x_train, y_train, x_test, y_test : 训练/测试数据 (DataFrame 或 numpy)
    l1_ratio   : Elastic Net 的 L1 混合比搜索范围
    top_n      : 取前 N 个候选子式进 Elastic Net

    返回 dict:
        coefs     : 各子式的系数（0 表示被剔除）
        selected  : 被选中的子式列表 [(子式, 系数), ...]
        formula   : 最终公式字符串
        mse, r2   : 测试集上的表现
    """

    import pandas as pd

    # 准备数据
    X_tr = x_train.values if isinstance(x_train, pd.DataFrame) else np.asarray(x_train)
    y_tr = y_train.values.ravel() if hasattr(y_train, "values") else np.asarray(y_train).ravel()

    if x_test is not None:
        X_te = x_test.values if isinstance(x_test, pd.DataFrame) else np.asarray(x_test)
    if y_test is not None:
        y_te = y_test.values.ravel() if hasattr(y_test, "values") else np.asarray(y_test).ravel()

    feat_names = x_train.columns.tolist() if hasattr(x_train, "columns") else []
    col_map = {name: i for i, name in enumerate(feat_names)}

    # 取 top_n 候选，先构建矩阵做相关性去重
    cand = candidates[:top_n]
    X_f = _eval_candidates(cand, X_tr, col_map)

    # 合并高度相关子式（|r| > 0.95），保留 pattern_score 更高的
    corr = np.corrcoef(X_f, rowvar=False)
    drop_idx = set()
    n = len(cand)
    for i in range(n):
        if i in drop_idx:
            continue
        for j in range(i + 1, n):
            if j in drop_idx:
                continue
            if abs(corr[i, j]) > 0.95:
                # 保留 pattern_score 高的，低的踢掉
                score_i = cand[i].get("pattern_score", 0)
                score_j = cand[j].get("pattern_score", 0)
                loser = j if score_i >= score_j else i
                drop_idx.add(loser)

    if drop_idx:
        keep = [i for i in range(n) if i not in drop_idx]
        cand = [cand[i] for i in keep]
        X_f = X_f[:, keep]

    # 标准化 → 公平对待不同量级子式（不影响公式结构）
    scaler = StandardScaler()
    X_f_scaled = scaler.fit_transform(X_f)

    # Elastic Net 自动筛选
    model = ElasticNetCV(
        l1_ratio=list(l1_ratio),
        cv=5,
        max_iter=5000,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_f_scaled, y_tr)

    # 系数还原到原始量级
    means = scaler.mean_
    stds = scaler.scale_
    coefs_scaled = model.coef_
    intercept_scaled = model.intercept_

    coefs = coefs_scaled / stds
    intercept = intercept_scaled - np.sum(coefs_scaled * means / stds)

    # 被选中的子式
    selected = [
        (cand[i], coefs[i])
        for i in range(len(cand))
        if abs(coefs[i]) > 1e-6
    ]

    # 构建公式字符串
    terms = []
    if abs(intercept) > 1e-6:
        terms.append(f"{intercept:.6f}")

    for c, w in selected:
        if abs(w) < 1e-6:
            continue
        w_str = f"{w:.6f}"
        se = c["subexpr"]
        # 如果子式本身是加法，加括号
        if "+" in se or "-" in se:
            se = f"({se})"
        terms.append(f"{w_str}*{se}")

    formula_str = " + ".join(terms) if terms else "0"

    # 测试集评估
    result = {
        "coefs": coefs.tolist(),
        "selected": [(c["subexpr"], round(w, 6)) for c, w in selected],
        "formula": formula_str,
        "intercept": round(float(intercept), 6),
        "mse": None,
        "r2": None,
    }

    if x_test is not None and y_test is not None:
        X_f_test = _eval_candidates(cand, X_te, col_map)
        X_f_test_scaled = scaler.transform(X_f_test)
        y_pred = model.predict(X_f_test_scaled)
        result["mse"] = round(float(mean_squared_error(y_te, y_pred)), 4)
        result["r2"] = round(float(r2_score(y_te, y_pred)), 4)
        result["y_pred"] = y_pred

    return result


def print_selection(result: dict):
    """打印 Elastic Net 筛选结果"""
    print(f"\n{'='*70}")
    print(f"Elastic Net 筛选结果")
    print(f"{'='*70}")
    print(f"截距: {result['intercept']}")
    print(f"\n选中 {len(result['selected'])} 项:")
    for se, w in result["selected"]:
        print(f"  {w:+.6f}  ×  {se}")
    print(f"\n最终公式:")
    print(f"  {result['formula']}")
    if result["mse"] is not None:
        print(f"\n测试集: MSE={result['mse']:.4f}  R²={result['r2']:.4f}")
