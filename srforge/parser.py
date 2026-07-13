from itertools import count
from .models import Formula, FormulaNode

def parse_formulas(formulas):
    all_nodes = []
    
    # 全局节点编号生成器
    node_counter = count()
    
    for formula in formulas:
        all_nodes.extend(
            parse_formula(formula,node_counter)
        )
    return all_nodes

def _walk(
    expr,
    formula,
    parent_id,
    depth,
    path,
    nodes,
    node_counter
):
    """
    递归遍历SymPy表达式树，构建FormulaNode列表
    """
    # 构建节点
    node_id = next(node_counter)
    node = FormulaNode(
        node_id = node_id,
        formula_id = formula.formula_id,
        expr = expr,
        parent_id = parent_id,
        depth = depth,
        path = path + (node_id,),
        run_id = formula.run_id,
    )
    nodes.append(node)
    
    # 遍历子节点
    for child in expr.args:
        _walk(
            expr = child,
            formula = formula,
            parent_id = node_id,
            depth = depth + 1,
            path = node.path,
            nodes = nodes,
            node_counter = node_counter,
        )
    
    

def parse_formula(formula:Formula, node_counter) -> list[FormulaNode]:
    """
    解析公式 Formumla -> FormulaNode列表
    """
    nodes :list[FormulaNode] = []
    
    # 全局节点编号生成器
    _walk(
        expr = formula.expr,
        formula = formula,
        parent_id = None,
        depth = 0,
        path = (),
        nodes = nodes,
        node_counter = node_counter,
    )
    
    return nodes