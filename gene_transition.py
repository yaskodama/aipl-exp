"""
gene_transition.py — .ga ログと個体ゲノム (.aice = HTML) を読み、
                     世代を通した遺伝子の移り変わりを分析する。

2 つの "遺伝子" レベルで分析:
    1. 関数レベル: <script> 内の関数定義（named / arrow / methods）
       → どの実装パターンが世代を超えて生き残るか（building block hypothesis）
    2. data-testid レベル: アプリの "機能マニフェスト"
       → どの機能がいつ集団内で固定（fixation）したか

使い方:
    python gene_transition.py runs/todo_s42.ga --out figs/

依存:
    pip install numpy matplotlib
"""

from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# モジュール変数: AST 抽出器の振る舞いを切り替える
_USE_AST_PARSER = False
_AST_TOOL = Path(__file__).parent / "tools" / "parse_funcs.mjs"

matplotlib.rcParams["font.family"] = [
    "Hiragino Sans", "Hiragino Maru Gothic Pro", "Yu Gothic",
    "Meiryo", "Noto Sans CJK JP", "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False


# ============================================================
# .ga パーサ（gene_transition.py 用に最小限）
# ============================================================
@dataclass
class Indiv:
    id: str; gen: int; op: str; fitness: float
    parents: list[str]; agent: str = ""; call: str = ""
    genome_ref: str = ""; notes: str = ""


@dataclass
class Run:
    path: Path
    meta: dict = field(default_factory=dict)
    indivs: list[Indiv] = field(default_factory=list)
    summaries: list[dict] = field(default_factory=list)


def _tokenize(line: str) -> list[str]:
    out, buf, q = [], "", False
    for i, ch in enumerate(line):
        if q:
            buf += ch
            if ch == '"' and line[i - 1] != "\\":
                q = False
        elif ch == '"':
            buf += ch; q = True
        elif ch.isspace():
            if buf: out.append(buf); buf = ""
        else:
            buf += ch
    if buf: out.append(buf)
    return out


def _val(v: str):
    if v.startswith('"') and v.endswith('"'): return v[1:-1]
    if v.startswith("[") and v.endswith("]"):
        return [s.strip() for s in v[1:-1].split(",") if s.strip()]
    return v


def parse_ga(path: Path) -> Run:
    run = Run(path=path)
    cur_gen = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("#"): continue
        parts = _tokenize(ln)
        head = parts[0]
        kv = {}
        for p in parts[1:]:
            if "=" not in p: continue
            k, v = p.split("=", 1)
            kv[k] = _val(v)
        if head == "META":
            run.meta.update(kv)
        elif head == "GEN":
            cur_gen = int(parts[1])
        elif head == "INDIV":
            par = kv.get("parents", [])
            if isinstance(par, str): par = [p for p in par.split(",") if p]
            run.indivs.append(Indiv(
                id=kv["id"], gen=cur_gen, op=kv.get("op", ""),
                fitness=float(kv["fitness"]), parents=par,
                agent=kv.get("agent", ""), call=kv.get("call", ""),
                genome_ref=kv.get("genome_ref", ""), notes=kv.get("notes", ""),
            ))
        elif head == "SUMMARY":
            run.summaries.append({k: (float(v) if k != "gen" else int(v))
                                  for k, v in kv.items() if isinstance(v, str)})
    return run


# ============================================================
# ゲノム読み込み（パスをロバストに解決）
# ============================================================
def resolve_genome(genome_ref: str, ga_path: Path) -> Path | None:
    if not genome_ref: return None
    p = Path(genome_ref)
    if p.is_absolute() and p.exists(): return p
    for base in [ga_path.parent, ga_path.parent.parent, Path.cwd()]:
        c = base / p
        if c.exists(): return c
    return None


def load_genome(ind: Indiv, ga_path: Path) -> str:
    p = resolve_genome(ind.genome_ref, ga_path)
    if p is None: return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ============================================================
# 関数抽出（balanced brace ベース）
# ============================================================
SCRIPT_RE = re.compile(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>", re.IGNORECASE)

# function NAME(...) {
FUNC_DECL_RE = re.compile(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*")

# const/let/var NAME = (...) => {   /   = function(...) {
ARROW_RE = re.compile(
    r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>\s*"
)
ASSIGN_FUNC_RE = re.compile(
    r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s*)?function\s*\([^)]*\)\s*"
)


def _balance_brace(code: str, start: int) -> int | None:
    """`start` が '{' を指しているとき、対応する '}' の直後の位置を返す。"""
    if start >= len(code) or code[start] != "{":
        return None
    depth = 0
    i = start
    in_str = None  # '"' or "'" or "`"
    esc = False
    while i < len(code):
        ch = code[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == in_str: in_str = None
        else:
            if ch in '"\'`': in_str = ch
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: return i + 1
        i += 1
    return None


def _next_open_brace(code: str, after: int) -> int | None:
    i = after
    while i < len(code) and code[i].isspace():
        i += 1
    if i < len(code) and code[i] == "{":
        return i
    return None


def extract_scripts(html: str) -> str:
    return "\n".join(SCRIPT_RE.findall(html))


def extract_via_ast(html: str) -> tuple[dict[str, str], set[str]]:
    """Node + acorn を呼んで AST ベース抽出。失敗時は regex にフォールバック。"""
    try:
        proc = subprocess.run(
            ["node", str(_AST_TOOL)],
            input=html, capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            return extract_functions_regex(extract_scripts(html)), extract_testids(html)
        data = json.loads(proc.stdout)
        funcs: dict[str, str] = {}
        for entry in data.get("functions", []):
            name = entry.get("name")
            body = entry.get("body_normalized", "")
            if name and body:
                funcs[name] = body
        tids = set(data.get("testids", []))
        return funcs, tids
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return extract_functions_regex(extract_scripts(html)), extract_testids(html)


def extract_functions_regex(code: str) -> dict[str, str]:
    """関数名 → 正規化済み本体テキスト。重複名は最後勝ち。"""
    found: dict[str, str] = {}
    seen_ranges: list[tuple[int, int]] = []

    def within_seen(pos: int) -> bool:
        return any(a <= pos < b for a, b in seen_ranges)

    for pat in (FUNC_DECL_RE, ARROW_RE, ASSIGN_FUNC_RE):
        for m in pat.finditer(code):
            if within_seen(m.start()): continue
            ob = _next_open_brace(code, m.end())
            if ob is None: continue
            cb = _balance_brace(code, ob)
            if cb is None: continue
            body = code[ob + 1 : cb - 1]
            norm = re.sub(r"\s+", " ", body).strip()
            if not norm: continue
            found[m.group(1)] = norm
            seen_ranges.append((m.start(), cb))
    return found


def extract_functions(code_or_html: str) -> dict[str, str]:
    """analyze() からの統一エントリ。--parser ast が指定されていれば AST 経由。
    AST 版は HTML 全体を必要とするので、analyze() からは HTML を渡す。"""
    if _USE_AST_PARSER:
        funcs, _tids = extract_via_ast(code_or_html)
        return funcs
    return extract_functions_regex(extract_scripts(code_or_html))


def body_hash(body: str) -> str:
    return hashlib.md5(body.encode("utf-8")).hexdigest()[:8]


# ============================================================
# data-testid 抽出（機能マニフェスト）
# ============================================================
TESTID_RE = re.compile(r'data-testid\s*=\s*[\'"]([A-Za-z_][\w-]*)[\'"]')


def extract_testids(html: str) -> set[str]:
    return set(TESTID_RE.findall(html))


# ============================================================
# 分析本体
# ============================================================
@dataclass
class GeneStats:
    # gen -> name -> 集団内 in-individual 出現数（同個体内で重複は 1 として数える）
    gen_func_pop: dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    gen_testid_pop: dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    gen_size: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # name -> {body_hash}（同名関数の本体バリアント数）
    func_body_variants: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    # 個体ごとの抽出結果（後段で使う）
    indiv_funcs: dict[str, set[str]] = field(default_factory=dict)
    indiv_testids: dict[str, set[str]] = field(default_factory=dict)


def analyze(run: Run) -> GeneStats:
    s = GeneStats()
    for ind in run.indivs:
        html = load_genome(ind, run.path)
        if not html: continue
        s.gen_size[ind.gen] += 1
        # extract_functions は HTML 全体を受け取り、内部で必要に応じて
        # AST か regex かを切り替える
        funcs = extract_functions(html)
        s.indiv_funcs[ind.id] = set(funcs.keys())
        for name, body in funcs.items():
            s.gen_func_pop[ind.gen][name] += 1
            s.func_body_variants[name].add(body_hash(body))
        tids = extract_testids(html)
        s.indiv_testids[ind.id] = tids
        for tid in tids:
            s.gen_testid_pop[ind.gen][tid] += 1
    return s


def fixation_gen(gen_pop: dict[int, Counter], gen_size: dict[int, int], name: str,
                 thresh: float = 1.0) -> int | None:
    """関数 / testid が集団内に thresh の割合で初めて存在した世代。"""
    for g in sorted(gen_size):
        n = gen_size[g] or 1
        if gen_pop[g].get(name, 0) / n >= thresh:
            return g
    return None


def find_fitness_jumps(run: Run, stats: GeneStats, delta_threshold: float = 0.15) -> list[dict]:
    """親→子で Δfitness > 閾値 のイベントを、関数・testid 差分付きで返す。"""
    by_id = {i.id: i for i in run.indivs}
    out = []
    for ind in run.indivs:
        if not ind.parents: continue
        parent_fits = [by_id[pid].fitness for pid in ind.parents if pid in by_id]
        if not parent_fits: continue
        pmax = max(parent_fits)
        delta = ind.fitness - pmax
        if delta < delta_threshold: continue
        child_f = stats.indiv_funcs.get(ind.id, set())
        child_t = stats.indiv_testids.get(ind.id, set())
        parent_f = set().union(*(stats.indiv_funcs.get(p, set()) for p in ind.parents))
        parent_t = set().union(*(stats.indiv_testids.get(p, set()) for p in ind.parents))
        out.append({
            "child": ind.id, "gen": ind.gen, "op": ind.op,
            "parents": ind.parents,
            "delta_fitness": round(delta, 4),
            "child_fitness": round(ind.fitness, 4),
            "added_funcs":    sorted(child_f - parent_f),
            "removed_funcs":  sorted(parent_f - child_f),
            "added_testids":  sorted(child_t - parent_t),
            "removed_testids":sorted(parent_t - child_t),
        })
    return out


# ============================================================
# 描画
# ============================================================
def plot_heatmap(gen_pop: dict[int, Counter], gen_size: dict[int, int],
                 title: str, out: Path, top_k: int = 20):
    gens = sorted(gen_size.keys())
    if not gens:
        print(f"  ({title}: データなし)"); return
    total = Counter()
    for g in gens:
        for name, c in gen_pop[g].items(): total[name] += c
    top = [n for n, _ in total.most_common(top_k)]
    if not top:
        print(f"  ({title}: 上位アイテムなし)"); return

    mat = np.zeros((len(top), len(gens)))
    for i, name in enumerate(top):
        for j, g in enumerate(gens):
            mat[i, j] = gen_pop[g].get(name, 0) / max(gen_size[g], 1)

    h = max(4, 0.32 * len(top))
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(gens) + 4), h))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(gens))); ax.set_xticklabels([f"g{g}" for g in gens])
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top)
    ax.set_xlabel("世代")
    ax.set_title(title)
    for i in range(len(top)):
        for j in range(len(gens)):
            v = mat[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.55 else "#333", fontsize=8)
    fig.colorbar(im, ax=ax, label="集団内出現率")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def plot_fitness_jumps_table(jumps: list[dict], out: Path):
    if not jumps:
        print("  (突破点なし — Δfitness 閾値を下げて再実行を検討)"); return
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.5 * len(jumps)))
    ax.axis("off")
    rows = []
    for j in jumps:
        rows.append([
            f"g{j['gen']}",
            j["child"],
            j["op"],
            f"+{j['delta_fitness']:.3f}",
            (", ".join(j["added_funcs"])    or "—")[:30],
            (", ".join(j["added_testids"])  or "—")[:30],
        ])
    table = ax.table(
        cellText=rows,
        colLabels=["gen", "個体", "op", "Δfit", "追加関数", "追加 testid"],
        loc="center", cellLoc="left",
    )
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("Fitness 突破点（親→子の差分）", pad=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ============================================================
# 出力
# ============================================================
def write_overlay(run: Run, stats: GeneStats, ga_path: Path, out_dir: Path) -> Path:
    """ga-viewer.html が読み込む overlay JSON を出力する。
    同じ stem の .overlay.json を .ga の隣にも書く（ビューアの自動ロード用）。"""
    by_id = {i.id: i for i in run.indivs}
    individuals = {}
    for ind in run.indivs:
        individuals[ind.id] = {
            "functions": sorted(stats.indiv_funcs.get(ind.id, set())),
            "testids":   sorted(stats.indiv_testids.get(ind.id, set())),
        }
    # 各機能の初出個体（最も古い世代の最初の保持者）
    def first_owner(kind: str, name: str) -> str | None:
        store = stats.indiv_funcs if kind == "func" else stats.indiv_testids
        best = None
        for ind in run.indivs:
            if name in store.get(ind.id, set()):
                if best is None or ind.gen < by_id[best].gen:
                    best = ind.id
        return best
    func_names: set[str] = set()
    for s in stats.indiv_funcs.values(): func_names |= s
    tid_names: set[str] = set()
    for s in stats.indiv_testids.values(): tid_names |= s
    func_first = {n: first_owner("func", n) for n in func_names}
    tid_first  = {n: first_owner("tid",  n) for n in tid_names}
    # 革新エッジ: 子が持つ機能のうち、親 (=parents の union) が持たない物
    innovation_edges = []
    for ind in run.indivs:
        if not ind.parents: continue
        parent_funcs = set().union(*(stats.indiv_funcs.get(p, set()) for p in ind.parents))
        parent_tids  = set().union(*(stats.indiv_testids.get(p, set()) for p in ind.parents))
        child_funcs  = stats.indiv_funcs.get(ind.id, set())
        child_tids   = stats.indiv_testids.get(ind.id, set())
        new_f = sorted(child_funcs - parent_funcs)
        new_t = sorted(child_tids  - parent_tids)
        if new_f or new_t:
            innovation_edges.append({
                "child": ind.id, "parents": ind.parents,
                "new_funcs": new_f, "new_testids": new_t,
            })

    payload = {
        "version": 1,
        "ga_file": ga_path.name,
        "individuals": individuals,
        "function_first_appearance": func_first,
        "testid_first_appearance":   tid_first,
        "innovation_edges": innovation_edges,
    }
    out_main = out_dir / "lineage_overlay.json"
    out_main.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # .ga の隣にも置く（ga-viewer.html が同名 .overlay.json を自動探索する）
    sidecar = ga_path.with_suffix(ga_path.suffix + ".overlay.json")
    sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {out_main}")
    print(f"  → {sidecar}  (ga-viewer.html 自動ロード用)")
    return out_main


def report(run: Run, stats: GeneStats, jumps: list[dict], out_dir: Path):
    # 関数の fixation
    func_fix = {}
    for name in stats.func_body_variants:
        func_fix[name] = {
            "fixation_gen": fixation_gen(stats.gen_func_pop, stats.gen_size, name, 1.0),
            "first_appearance_gen": fixation_gen(stats.gen_func_pop, stats.gen_size, name, 1e-9),
            "body_variants": len(stats.func_body_variants[name]),
        }
    testid_fix = {}
    all_tids = set()
    for c in stats.gen_testid_pop.values(): all_tids |= set(c.keys())
    for tid in all_tids:
        testid_fix[tid] = {
            "fixation_gen": fixation_gen(stats.gen_testid_pop, stats.gen_size, tid, 1.0),
            "first_appearance_gen": fixation_gen(stats.gen_testid_pop, stats.gen_size, tid, 1e-9),
        }

    payload = {
        "run": str(run.path),
        "meta": run.meta,
        "n_indivs": len(run.indivs),
        "generations": sorted(stats.gen_size.keys()),
        "population_per_gen": dict(stats.gen_size),
        "function_fixation": func_fix,
        "testid_fixation": testid_fix,
        "fitness_jumps": jumps,
    }
    p = out_dir / "gene_transition_report.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {p}")

    # 標準出力サマリ
    print("\n--- 関数 building block ---")
    rows = sorted(func_fix.items(), key=lambda kv: (kv[1]["fixation_gen"] is None, kv[1]["fixation_gen"] or 999))
    for name, info in rows[:15]:
        fx = "—" if info["fixation_gen"] is None else f"g{info['fixation_gen']}"
        fa = f"g{info['first_appearance_gen']}" if info["first_appearance_gen"] is not None else "—"
        print(f"  {name:<22} 初出={fa:<5} 固定={fx:<5} 本体バリアント={info['body_variants']}")

    print("\n--- data-testid 機能マニフェスト ---")
    rows = sorted(testid_fix.items(), key=lambda kv: (kv[1]["fixation_gen"] is None, kv[1]["fixation_gen"] or 999))
    for tid, info in rows[:15]:
        fx = "—" if info["fixation_gen"] is None else f"g{info['fixation_gen']}"
        fa = f"g{info['first_appearance_gen']}" if info["first_appearance_gen"] is not None else "—"
        print(f"  {tid:<22} 初出={fa:<5} 固定={fx:<5}")

    print("\n--- Fitness 突破点（Δ>=0.15） ---")
    for j in jumps:
        print(f"  g{j['gen']} {j['child']:<6} op={j['op']:<10} Δ={j['delta_fitness']:+.3f}"
              f"  +funcs={j['added_funcs']}  +testids={j['added_testids']}")


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("ga_file", help=".ga ファイルパス")
    p.add_argument("--out", default="figs", help="図と JSON の出力ディレクトリ")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--jump-threshold", type=float, default=0.15)
    p.add_argument("--parser", choices=["regex", "ast"], default="regex",
                   help="関数抽出器: regex（既定、Node 不要）/ ast（acorn 必須、より高精度）")
    args = p.parse_args()

    global _USE_AST_PARSER
    _USE_AST_PARSER = (args.parser == "ast")
    if _USE_AST_PARSER:
        if not _AST_TOOL.exists():
            raise SystemExit(f"AST tool not found: {_AST_TOOL}")
        # Node + acorn の有無を 1 回だけ事前確認
        try:
            test = subprocess.run(["node", str(_AST_TOOL)],
                                  input="<html></html>", capture_output=True,
                                  text=True, timeout=10)
            if test.returncode != 0:
                print(f"warn: AST parser fallback to regex (node returned {test.returncode})")
                print(f"      stderr: {test.stderr[:200]}")
                _USE_AST_PARSER = False
        except FileNotFoundError:
            raise SystemExit("node not found. AST parser requires Node.js. `npm install` in aipl-exp/ first.")

    ga = Path(args.ga_file).resolve()
    if not ga.exists():
        raise SystemExit(f"not found: {ga}")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {ga} ...")
    run = parse_ga(ga)
    print(f"  {len(run.indivs)} 個体 / {len(run.summaries)} 世代")
    print(f"analyzing genomes ...")
    stats = analyze(run)
    print(f"  抽出: {sum(len(v) for v in stats.indiv_funcs.values())} 関数オカレンス, "
          f"{sum(len(v) for v in stats.indiv_testids.values())} testid オカレンス")
    jumps = find_fitness_jumps(run, stats, args.jump_threshold)

    print("\n=== 図 ===")
    plot_heatmap(stats.gen_func_pop, stats.gen_size,
                 "関数の集団内出現率（building block 持続）",
                 out_dir / "functions_heatmap.png", args.top_k)
    plot_heatmap(stats.gen_testid_pop, stats.gen_size,
                 "data-testid（機能マニフェスト）の集団内出現率",
                 out_dir / "testids_heatmap.png", args.top_k)
    plot_fitness_jumps_table(jumps, out_dir / "fitness_jumps.png")

    print("\n=== レポート ===")
    report(run, stats, jumps, out_dir)

    print("\n=== ga-viewer.html 用 overlay ===")
    write_overlay(run, stats, ga, out_dir)


if __name__ == "__main__":
    main()
