# srforge

多轮符号回归公式的结构化分析与公式锻造。

## 前置：多轮 PySR 训练

```python
from pysr import PySRRegressor

def train(x_train, y_train, n):
    results = []
    for seed in range(n):
        model = PySRRegressor(
            # ... 参数按需配置 ...
            random_state = seed,
            deterministic = True,
        )
        model.fit(x_train, y_train)
        results.append(model.equations_)
    return results

equations_dfs = train(x_train, y_train, 40)
```

## 路径 1：一键分析

```python
from srforge import build_formulas, save_equations, load_equations, full_analysis

# 保存
save_equations(equations_dfs)                     # ipynb 里存

# 分析
dfs = load_equations()
formulas = build_formulas(dfs)
full_analysis(formulas)                           # 终端输出 + 浏览器打开报告
```

`full_analysis` 返回的字典：

```python
result["reports"]  → Pattern 报告列表（给 export_html 用）
result["scored"]   → 公式排名列表（给 extract_candidates 用）
result["patterns"] → 原始 Pattern 对象（给 extract_candidates 用）
result["context"]  → 全局统计（Run 总数、公式总数、最大频率）
```

## 路径 2：公式锻造（ElasticNet）

```python
from srforge import extract_candidates, elastic_select

result = full_analysis(formulas)
candidates = extract_candidates(formulas, result["scored"],
                                result["patterns"], result["context"],
                                x_filter=x_train)
sel = elastic_select(candidates, x_train, y_train, x_test, y_test)
# → 最终公式 + 测试集 R²/MSE
```

## 路径 3：二次搜索（PySR Round 2）

```python
from srforge import round2_forge, expand_feats

r2 = round2_forge(result, formulas, x_train, y_train, x_test, y_test,
                  train_fn=train)
# 可选参数: save_path="equations_round2.pkl", output_html="report_round2.html"

# r2["scored"]   → Round 2 公式排名
# r2["feat_map"] → 用于 expand_feats 还原表达式
# r2["formulas"] → Round 2 的 Formula 对象（可继续 extract_candidates）
```

## 路径 4：稳定性验证

```python
from srforge import cross_validate, print_cv

df = cross_validate(equation, X, y)
print_cv(df)
# 输出每轮 seed 的 R²/MSE + 汇总统计
```

## 路径 5：公式安全检查

```python
from srforge import check_formula

warnings = check_formula(equation, X)
# 空列表 = 安全，否则返回问题描述
```

## 函数速查

| 函数 | 步骤 | 说明 |
|---|---|---|
| `save_equations` / `load_equations` | 保存 | DataFrame 列表 ↔ pkl |
| `build_formulas` | 构建 | DataFrame → Formula 对象 |
| `full_analysis` | 分析 | 一键：报告 + 评分 + HTML |
| `quick_score` | 分析 | 仅公式排名 |
| `extract_candidates` | 锻造 | 提取共识子式池 |
| `elastic_select` | 锻造 | ElasticNet 筛选 → 最终公式 |
| `round2_forge` | 二次搜索 | 提取子式 → 新特征 → PySR Round 2 |
| `augment_features` | 二次搜索 | 子式 → 新特征列 |
| `expand_feats` | 二次搜索 | feat_N → 原始表达式 |
| `cross_validate` | 验证 | 跨 seed 稳定性评估 |
| `check_formula` | 验证 | 公式安全检查 |
| `export_html` | 报告 | 单独导出 HTML |

## 报告与评分

HTML 报告包含三部分：

- **汇总卡片** — Pattern 总数、Run 数、公式数、最高分
- **Pattern 总览** — 每个结构模式的评分、覆盖率、频率（进度条 + 颜色标签）
- **Slot 详情** — 每个结构位置上的变量分布（条形图）

### 公式最终分

```
final = 0.6 × loss_score + 0.1 × complexity_score + 0.3 × structure_score
```

- **loss_score**：`1 − (loss − min) / (max − min)`，越低得分越高
- **complexity_score**：同上
- **structure_score**：公式所有非叶节点的 slot 匹配分平均值

### Pattern 评分（报告中显示）

```
total = 0.4 × Run覆盖 + 0.3 × Formula覆盖 + 0.2 × 变量稳定性 + 0.1 × 归一化频率
```

### Slot 匹配分

对比节点每个子位置的实际变量与 Pattern 共识分布。该变量在 slot 分布中的占比 = 该位置得分，所有位置取平均。不存在于分布中的变量得 0。
