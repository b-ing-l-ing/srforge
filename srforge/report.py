"""
报告导出模块
将 run_analyzer 的输出导出为自包含 HTML 文件，
浏览器打开即可查看，无需额外依赖。
"""

import webbrowser
from pathlib import Path

# -----------------------------------------------------
# HTML 模板
# -----------------------------------------------------

_CSS = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #333; padding: 24px; }
  h1 { font-size: 22px; margin-bottom: 8px; }
  .summary { display: flex; gap: 16px; margin: 16px 0 24px; flex-wrap: wrap; }
  .summary .card { background: #fff; border-radius: 8px; padding: 14px 20px; min-width: 120px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .summary .card .num { font-size: 28px; font-weight: 700; color: #2563eb; }
  .summary .card .label { font-size: 13px; color: #888; margin-top: 2px; }

  table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  th { background: #f0f0f0; text-align: left; padding: 10px 14px; font-size: 13px; font-weight: 600; }
  td { padding: 10px 14px; font-size: 13px; border-top: 1px solid #eee; }
  tr:hover td { background: #fafbff; }
  .sig { font-family: "Cascadia Code", "Fira Code", monospace; font-size: 12px; color: #555; word-break: break-all; }
  .bar-wrap { background: #eee; border-radius: 4px; height: 14px; width: 80px; overflow: hidden; display: inline-block; vertical-align: middle; }

  .tag { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
  .tag-high { background: #dcfce7; color: #166534; }
  .tag-mid { background: #fef9c3; color: #854d0e; }
  .tag-low { background: #fee2e2; color: #991b1b; }

  .detail { margin-top: 24px; }
  .detail h2 { font-size: 17px; margin-bottom: 16px; }
  .slot-card { background: #fff; border-radius: 8px; padding: 16px 20px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .slot-card h3 { font-size: 14px; margin-bottom: 10px; color: #555; }
  .slot-bar { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
  .slot-bar .name { width: 70px; font-size: 12px; text-align: right; font-weight: 500; }
  .slot-bar .bar-bg { flex: 1; max-width: 260px; background: #eee; border-radius: 3px; height: 16px; overflow: hidden; }
  .slot-bar .bar-inner { display: block; height: 100%; border-radius: 3px; }
  .slot-bar .pct { width: 44px; font-size: 12px; color: #666; }
  .slot-meta { display: flex; gap: 24px; margin-top: 8px; font-size: 12px; color: #888; }

  .toc { margin-bottom: 24px; }
  .toc a { display: inline-block; margin: 0 12px 6px 0; color: #2563eb; font-size: 13px; font-family: monospace; }

  .formula-section { margin-bottom: 32px; }
  .formula-section h2 { font-size: 17px; margin-bottom: 12px; }
  .formula-table .eq { font-family: "Cascadia Code","Fira Code",monospace; font-size: 12px; word-break: break-all; }
  .formula-table .sc { font-weight: 600; }
</style>
"""

_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head><meta charset="UTF-8"><title>srforge Report</title>
{css}
</head>
<body>

<h1>srforge Report</h1>

<div class="summary">
  <div class="card"><div class="num">{n_patterns}</div><div class="label">Patterns</div></div>
  <div class="card"><div class="num">{n_runs}</div><div class="label">Runs</div></div>
  <div class="card"><div class="num">{n_formulas}</div><div class="label">Formulas</div></div>
  <div class="card"><div class="num">{top_score}</div><div class="label">Top Score</div></div>
</div>

<div class="toc"><strong>快速跳转：</strong><br>{toc}</div>

{formula_section}

<h2>Pattern 总览</h2>
<table>
<thead><tr>
  <th>#</th><th>Signature</th><th>Type</th><th>Score</th><th>Stability</th><th>Run Cov</th><th>Formula Cov</th><th>Frequency</th>
</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>

<div class="detail">
<h2>Pattern 详情</h2>
{detail_sections}
</div>

</body></html>"""


# -----------------------------------------------------

def _score_tag(score: float) -> str:
    if score >= 0.8:
        return "tag tag-high"
    elif score >= 0.6:
        return "tag tag-mid"
    else:
        return "tag tag-low"


def _percent_bar(ratio: float) -> str:
    pct = round(ratio * 100)
    # 根据比例变色：高→绿，中→黄，低→红
    if pct >= 80:
        color = "#16a34a"   # 绿
    elif pct >= 50:
        color = "#eab308"   # 黄
    elif pct >= 20:
        color = "#f97316"   # 橙
    else:
        color = "#dc2626"   # 红
    return (
        f'<span class="bar-wrap">'
        f'<span style="height:100%;border-radius:4px;background:{color};'
        f'width:{pct}%;display:block"></span>'
        f'</span> {pct}%'
    )


def _build_toc(reports: list[dict]) -> str:
    """生成顶部快速跳转链接"""
    links = []
    for i, r in enumerate(reports, 1):
        sig_short = r["signature"]
        if len(sig_short) > 36:
            sig_short = sig_short[:33] + "..."
        links.append(f'<a href="#p{i}">{sig_short}</a>')
    return "".join(links)


def _build_table_rows(reports: list[dict]) -> str:
    rows = []
    for i, r in enumerate(reports, 1):
        s = r["score"]
        sc = s["total"]
        tag_cls = _score_tag(sc)
        tag_label = "★ 优先" if sc >= 0.8 else ("▲ 保留" if sc >= 0.6 else "· 参考")

        rows.append(
            f'<tr>'
            f'<td><a href="#p{i}">{i}</a></td>'
            f'<td><span class="sig">{r["signature"]}</span></td>'
            f'<td><span class="sig">{r.get("type", "-")}</span></td>'
            f'<td><span class="{tag_cls}">{tag_label} {sc:.3f}</span></td>'
            f'<td>{_percent_bar(s["stability"])}</td>'
            f'<td>{_percent_bar(s["run"])}</td>'
            f'<td>{_percent_bar(s["formula"])}</td>'
            f'<td>{_percent_bar(s["frequency"])}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def _build_slot_bars(distribution: dict) -> str:
    """给一个 slot 的 distribution 画水平条形图"""
    total = sum(distribution.values())
    if total == 0:
        return ""

    # 按数量降序
    items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)

    # 颜色：第一是蓝色，其余渐浅
    colors = ["#2563eb", "#60a5fa", "#93c5fd", "#bfdbfe"]

    bars = []
    for j, (var, count) in enumerate(items):
        ratio = count / total
        pct = round(ratio * 100)
        color = colors[j] if j < len(colors) else "#dbeafe"
        bars.append(
            f'<div class="slot-bar">'
            f'<span class="name">{var}</span>'
            f'<span class="bar-bg">'
            f'<span class="bar-inner" style="width:{pct}%;background:{color}"></span>'
            f'</span>'
            f'<span class="pct">{pct}%</span>'
            f'<span style="font-size:11px;color:#999">({count}次)</span>'
            f'</div>'
        )
    return "\n".join(bars)


def _build_detail_sections(reports: list[dict]) -> str:
    sections = []
    for i, r in enumerate(reports, 1):
        s = r["score"]
        sc = s["total"]
        tag_label = "★ 优先候选" if sc >= 0.8 else ("▲ 建议保留" if sc >= 0.6 else "· 作为参考")

        # slot 详情
        slot_html = ""
        slots_data = r.get("slot_manual_view") or r.get("slots", {})
        for slot_key, info in slots_data.items():
            dist = info.get("distribution", {})
            conf = info.get("confidence", 0)

            slot_html += (
                f'<div class="slot-card">'
                f'<h3>{slot_key}</h3>'
                f'{_build_slot_bars(dist)}'
                f'<div class="slot-meta">'
                f'<span>Confidence: {conf:.3f}</span>'
                f'<span>Top1: {info.get("top1", "-")} ({info.get("top1_ratio", 0):.2f})</span>'
                f'</div>'
                f'</div>'
            )

        if not slot_html:
            slot_html = '<p style="color:#999;font-size:13px">无 slot 信息</p>'

        sections.append(
            f'<div id="p{i}" style="margin-bottom:32px">'
            f'<h3 style="margin-bottom:6px">{i}. {r["signature"]}</h3>'
            f'<p style="font-size:13px;color:#666;margin-bottom:10px">'
            f'Score: {sc:.3f} &nbsp;|&nbsp; {tag_label} &nbsp;|&nbsp; '
            f'Run: {r["coverage"]["run"]["count"]}/{r["coverage"]["run"]["total"]} &nbsp;|&nbsp; '
            f'Formula: {r["coverage"]["formula"]["count"]}/{r["coverage"]["formula"]["total"]} &nbsp;|&nbsp; '
            f'Frequency: {r["frequency"]}'
            f'</p>'
            f'{slot_html}'
            f'</div>'
        )
    return "\n".join(sections)


def _build_formula_section(scored: list[dict]) -> str:
    """公式排名表格"""
    if not scored:
        return ""

    rows = []
    for i, r in enumerate(scored, 1):
        eq = r["equation"]
        fs = r["final_score"]
        tag_cls = "tag tag-high" if fs >= 0.8 else ("tag tag-mid" if fs >= 0.6 else "tag tag-low")
        rows.append(
            f'<tr>'
            f'<td>{i}</td>'
            f'<td class="eq">{eq}</td>'
            f'<td>{r["loss"]:.6f}</td>'
            f'<td>{r["complexity"]}</td>'
            f'<td class="sc">{r["structure_score"]:.3f}</td>'
            f'<td class="sc"><span class="{tag_cls}">{fs:.3f}</span></td>'
            f'</tr>'
        )

    rows_html = "\n".join(rows)
    return f"""
    <div class="formula-section">
    <h2>推荐公式</h2>
    <table class="formula-table">
    <thead><tr>
      <th>#</th><th>Equation</th><th>Loss</th><th>Comp</th><th>Struct</th><th>Final</th>
    </tr></thead>
    <tbody>
    {rows_html}
    </tbody>
    </table>
    </div>
    """


def export_html(reports: list[dict],
                output_path: str | Path = "report.html",
                open_browser: bool = True,
                scored: list[dict] | None = None):
    """
    将分析报告导出为 HTML 文件

    reports     : run_analyzer() 的返回值（list[dict]）
    output_path : 输出 HTML 路径
    open_browser: 是否自动用浏览器打开
    """

    # 汇总数据
    n_patterns = len(reports)
    if n_patterns == 0:
        raise ValueError("reports 为空，请先运行 run_analyzer()")

    n_runs = max(
        (r["coverage"]["run"]["total"] for r in reports),
        default=0
    )
    n_formulas = max(
        (r["coverage"]["formula"]["total"] for r in reports),
        default=0
    )
    top_score = max(r["score"]["total"] for r in reports)

    # 补充 type 字段（如果 report 里没有的话）
    for r in reports:
        r.setdefault("type", "-")

    html = _TEMPLATE.format(
        css=_CSS,
        n_patterns=n_patterns,
        n_runs=n_runs,
        n_formulas=n_formulas,
        top_score=f"{top_score:.3f}",
        toc=_build_toc(reports),
        formula_section=_build_formula_section(scored or []),
        table_rows=_build_table_rows(reports),
        detail_sections=_build_detail_sections(reports),
    )

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    print(f"报告已保存: {output_path.resolve()}")

    if open_browser:
        webbrowser.open(str(output_path.resolve()))
