from dataclasses import dataclass, field
from sympy import Expr

@dataclass
class Formula:
    """
    保存一条 PySR公式信息
    """
    formula_id: int         # 唯一编号
    run_id: int             # 来自第几次PySR
    expr: Expr              # 原始公式
    loss: float
    complexity: int

    @property
    def equation(self):
        """
        返回公式 字符串
        """
        return str(self.expr)

@dataclass
class FormulaNode:
    """
    SymPy表达式树中的一个节点
    """
    
    node_id: int          # 全局唯一编号
    formula_id: int       # 公式编号
    expr: Expr            # 当前 SymPy 节点(表达式)
    parent_id: int | None  # 父节点编号 可为空
    depth: int            # 节点深度(根节点0)
    run_id: int           # 来自第几批(如多轮pysr)
    path: tuple[int, ...] # 由根节点到本节点的路径(id,且包括自身)
    signature: str = field(default="")    # 结构标签
    
    @property
    def func(self):
        """节点类型,Add、Mul···"""
        return self.expr.func.__name__
    
    @property
    def is_leaf(self):
        """判断是否是叶子节点"""
        return len(self.expr.args) == 0
    
    @property
    def text(self):
        """节点对应的字符串"""
        return str(self.expr)
    
    
@dataclass
class Pattern:
    """
    一个结构模式 = signature + slot统计 + 出现频率
    """
    signature: str
    frequency: int
    slot_stats: dict[str, dict[str, int]]   # slot -> variable -> count
    type: str
    formula_ids: list[int] = field(default_factory=list)
    run_ids: list[int] = field(default_factory=list)
    
@dataclass
class AnalysisContext:
    """
    Analyzer 全局统计信息
    为 Pattern、Slot 等分析提供统一上下文
    """
    
    total_runs: int
    total_formulas: int
    max_frequency: int
    
    