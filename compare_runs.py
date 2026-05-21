"""
compare_runs.py — 複数試行をラベル別に集計して条件間比較する。

使い方:
    python compare_runs.py \
        --group cold runs/cold_*.ga \
        --group warm runs/warm_*.ga \
        --out comp/

出力:
    comp/
      ├── best_fitness_box.png      条件別 best fitness 箱ひげ図 + Mann-Whitney
      ├── ttt_box.png               目標到達世代の箱ひげ図
      ├── fitness_curves.png        条件別 中央値±IQR 収束曲線
      ├── building_block_timing.png 機能 (function/testid) の初出世代を条件比較
      └── compare_report.json       数値サマリ

依存: numpy, scipy, matplotlib
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

import gene_transition as gt

matplotlib.rcParams["font.family"] = [
    "Hiragino Sans", "Hiragino Maru Gothic Pro", "Yu Gothic",
    "Meiryo", "Noto Sans CJK JP", "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False


# ============================================================
# 集計
# ============================================================
@dataclass
class GroupResult:
    label: str
    n_runs: int = 0
    best_fitness: list[float] = field(default_factory=list)
    ttt: list[int] = field(default_factory=list)
    total_cost: list[float] = field(default_factory=list)
    summaries_per_run: list[list[dict]] = field(default_factory=list)
    # 機能名 → 各試行での初出世代（出ない試行は含めない）
    func_emerge: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    func_fixation: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    testid_emerge: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    testid_fixation: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))


def _safe_int(v) -> int | None:
    try:
        if v in (None, "-", ""): return None
        return int(v)
    except (ValueError, TypeError):
        return None


def _safe_float(v) -> float | None:
    try:
        if v in (None, "-", ""): return None
        return float(v)
    except (ValueError, TypeError):
        return None


def _ttt_from_summary(run: gt.Run, threshold: float) -> int | None:
    """SUMMARY 行から threshold 到達世代を post-hoc 計算（META が無い試行用）。"""
    for s in run.summaries:
        if s.get("best", 0.0) >= threshold:
            return int(s.get("gen", 0))
    return None


def aggregate(label: str, ga_paths: list[Path],
              ttt_threshold: float | None = None) -> GroupResult:
    g = GroupResult(label=label)
    for ga in ga_paths:
        try:
            run = gt.parse_ga(ga)
        except Exception as e:
            print(f"  ! parse failed {ga}: {e}", file=sys.stderr)
            continue
        g.n_runs += 1
        # 試行のメタからスカラー
        best = _safe_float(run.meta.get("best_fitness")) \
               or max((i.fitness for i in run.indivs), default=0.0)
        g.best_fitness.append(best)
        # TTT: --ttt-threshold 指定時は SUMMARY から post-hoc 計算
        if ttt_threshold is not None:
            ttt = _ttt_from_summary(run, ttt_threshold)
        else:
            ttt = _safe_int(run.meta.get("generations_to_target"))
        if ttt is not None: g.ttt.append(ttt)
        cost = _safe_float(run.meta.get("total_cost"))
        if cost is not None: g.total_cost.append(cost)
        g.summaries_per_run.append(run.summaries)
        # ゲノム解析
        s = gt.analyze(run)
        for name in s.func_body_variants:
            fa = gt.fixation_gen(s.gen_func_pop, s.gen_size, name, 1e-9)
            fx = gt.fixation_gen(s.gen_func_pop, s.gen_size, name, 1.0)
            if fa is not None: g.func_emerge[name].append(fa)
            if fx is not None: g.func_fixation[name].append(fx)
        all_tids = set()
        for c in s.gen_testid_pop.values():
            all_tids |= set(c.keys())
        for tid in all_tids:
            fa = gt.fixation_gen(s.gen_testid_pop, s.gen_size, tid, 1e-9)
            fx = gt.fixation_gen(s.gen_testid_pop, s.gen_size, tid, 1.0)
            if fa is not None: g.testid_emerge[tid].append(fa)
            if fx is not None: g.testid_fixation[tid].append(fx)
    return g


# ============================================================
# 統計
# ============================================================
def cliffs_delta(a, b) -> float:
    a, b = np.asarray(list(a)), np.asarray(list(b))
    if len(a) == 0 or len(b) == 0: return 0.0
    gt_cnt = sum(int(np.sum(x > b)) for x in a)
    lt_cnt = sum(int(np.sum(x < b)) for x in a)
    return (gt_cnt - lt_cnt) / (len(a) * len(b))


def pairwise(groups: list[GroupResult]) -> dict:
    out = {}
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i], groups[j]
            key = f"{a.label}_vs_{b.label}"
            block = {}
            if len(a.best_fitness) >= 2 and len(b.best_fitness) >= 2:
                u, p = stats.mannwhitneyu(a.best_fitness, b.best_fitness,
                                          alternative="two-sided")
                block["best_fitness"] = {
                    "median_a": float(np.median(a.best_fitness)),
                    "median_b": float(np.median(b.best_fitness)),
                    "U": float(u), "p": float(p),
                    "cliffs_delta": round(cliffs_delta(a.best_fitness, b.best_fitness), 3),
                }
            if len(a.ttt) >= 2 and len(b.ttt) >= 2:
                u, p = stats.mannwhitneyu(a.ttt, b.ttt, alternative="two-sided")
                block["ttt"] = {
                    "median_a": float(np.median(a.ttt)),
                    "median_b": float(np.median(b.ttt)),
                    "U": float(u), "p": float(p),
                    "cliffs_delta": round(cliffs_delta(a.ttt, b.ttt), 3),
                }
            out[key] = block
    return out


# ============================================================
# 描画
# ============================================================
def _box(ax, datas: list[list[float]], labels: list[str], ylabel: str, title: str):
    if not any(datas):
        ax.text(0.5, 0.5, "データなし", ha="center", va="center")
        ax.set_axis_off()
        return
    bp = ax.boxplot(datas, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor="#cfd5de", alpha=0.5))
    for i, d in enumerate(datas):
        ax.scatter([i + 1] * len(d), d, alpha=0.5, s=14, color="#3a6ea5")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")


def plot_box_summary(groups: list[GroupResult], out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    _box(ax,
         [g.best_fitness for g in groups],
         [f"{g.label}\n(n={g.n_runs})" for g in groups],
         "best fitness", "条件別 best fitness")
    fig.tight_layout(); fig.savefig(out_dir / "best_fitness_box.png", dpi=150); plt.close(fig)
    print(f"  → {out_dir/'best_fitness_box.png'}")

    fig, ax = plt.subplots(figsize=(6, 4))
    _box(ax,
         [g.ttt for g in groups],
         [f"{g.label}\n(n={len(g.ttt)})" for g in groups],
         "Time-To-Target (世代)", "目標到達までの世代数")
    fig.tight_layout(); fig.savefig(out_dir / "ttt_box.png", dpi=150); plt.close(fig)
    print(f"  → {out_dir/'ttt_box.png'}")


def plot_fitness_curves(groups: list[GroupResult], out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.tab10.colors
    for k, g in enumerate(groups):
        all_gens = sorted({s["gen"] for sums in g.summaries_per_run for s in sums})
        if not all_gens: continue
        mat = np.full((g.n_runs, len(all_gens)), np.nan)
        for i, sums in enumerate(g.summaries_per_run):
            d = {s["gen"]: s.get("best", 0.0) for s in sums}
            last = 0.0
            for j, gen in enumerate(all_gens):
                if gen in d: last = d[gen]
                mat[i, j] = last
        med = np.nanmedian(mat, axis=0)
        q25 = np.nanpercentile(mat, 25, axis=0)
        q75 = np.nanpercentile(mat, 75, axis=0)
        c = colors[k % 10]
        ax.fill_between(all_gens, q25, q75, alpha=0.18, color=c)
        ax.plot(all_gens, med, "-o", color=c, ms=4, lw=2,
                label=f"{g.label} (n={g.n_runs})")
    ax.set_xlabel("世代")
    ax.set_ylabel("best fitness（中央値 ± IQR）")
    ax.set_title("条件別の収束曲線")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_dir / "fitness_curves.png", dpi=150); plt.close(fig)
    print(f"  → {out_dir/'fitness_curves.png'}")


def plot_block_timing(groups: list[GroupResult], out_dir: Path, top_k: int = 12):
    """関数 + testid の初出世代を条件間で並べる（小さい方が早く獲得）。"""
    # ターゲット集合: 全条件で1回以上現れた上位 K（合計試行回数で並べる）
    score = defaultdict(int)
    for g in groups:
        for name, gens in g.func_emerge.items():
            score[("F", name)] += len(gens)
        for tid, gens in g.testid_emerge.items():
            score[("T", tid)] += len(gens)
    top = [k for k, _ in sorted(score.items(), key=lambda kv: -kv[1])[:top_k]]
    if not top:
        print("  (building block データなし)"); return

    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.4 * len(top)))
    colors = plt.cm.tab10.colors
    width = 0.8 / max(len(groups), 1)
    for k, g in enumerate(groups):
        ys, xs = [], []
        for kind, name in top:
            d = g.func_emerge if kind == "F" else g.testid_emerge
            vals = d.get(name, [])
            ys.extend([f"{kind}: {name}"] * len(vals))
            xs.extend(vals)
        if not xs: continue
        y_pos_map = {(kind+': '+name): i for i, (kind, name) in enumerate(top)}
        ypos = np.array([y_pos_map[y] for y in ys])
        jitter = (k - len(groups) / 2) * width + width / 2
        ax.scatter(xs, ypos + jitter, s=30, color=colors[k % 10],
                   alpha=0.75, edgecolors="white", linewidths=0.5,
                   label=f"{g.label}")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{kind}: {name}" for kind, name in top])
    ax.set_xlabel("初出世代")
    ax.set_title("機能 (F: 関数 / T: testid) の初出世代の条件比較")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(out_dir / "building_block_timing.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir/'building_block_timing.png'}")


# ============================================================
# レポート
# ============================================================
def summarize(groups: list[GroupResult], comp: dict) -> dict:
    out = {"groups": [], "comparisons": comp}
    for g in groups:
        bf = g.best_fitness
        ttt = g.ttt
        out["groups"].append({
            "label": g.label,
            "n_runs": g.n_runs,
            "best_fitness": {
                "median": float(np.median(bf)) if bf else None,
                "mean":   float(np.mean(bf))   if bf else None,
                "std":    float(np.std(bf))    if bf else None,
                "values": bf,
            },
            "ttt": {
                "median": float(np.median(ttt)) if ttt else None,
                "n_reached": len(ttt),
                "values": ttt,
            },
            "n_functions_observed": len(g.func_emerge),
            "n_testids_observed":   len(g.testid_emerge),
        })
    return out


def print_summary(report: dict):
    print("\n--- 条件別サマリ ---")
    for g in report["groups"]:
        bf = g["best_fitness"]
        print(f"  [{g['label']}] n={g['n_runs']}"
              f"  best={bf['median']:.3f}±{bf['std']:.3f}（中央値±SD）"
              f"  TTT(中央値)={g['ttt']['median']}"
              f"  到達={g['ttt']['n_reached']}/{g['n_runs']}")
    print("\n--- 条件間比較（Mann-Whitney U + Cliff's δ） ---")
    for key, blk in report["comparisons"].items():
        for metric, vals in blk.items():
            print(f"  {key} / {metric}:"
                  f"  med_a={vals['median_a']:.3f}  med_b={vals['median_b']:.3f}"
                  f"  U={vals['U']:.1f}  p={vals['p']:.4f}  δ={vals['cliffs_delta']:+.3f}")


# ============================================================
# 引数パーサ: --group LABEL FILE...
# ============================================================
def parse_groups(argv: list[str]) -> tuple[list[tuple[str, list[Path]]], list[str]]:
    """argv から --group LABEL PATTERNS... を抽出。残りを返す。"""
    groups: list[tuple[str, list[Path]]] = []
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--group":
            if i + 1 >= len(argv):
                raise SystemExit("--group requires a label and patterns")
            label = argv[i + 1]
            patterns: list[str] = []
            j = i + 2
            while j < len(argv) and not argv[j].startswith("--"):
                patterns.append(argv[j])
                j += 1
            files: list[Path] = []
            for pat in patterns:
                expanded = glob(pat)
                if not expanded and Path(pat).exists():
                    expanded = [pat]
                files.extend(Path(p) for p in expanded)
            groups.append((label, files))
            i = j
        else:
            remaining.append(argv[i])
            i += 1
    return groups, remaining


def main():
    raw = sys.argv[1:]
    group_specs, rest = parse_groups(raw)
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="comp")
    p.add_argument("--ttt-threshold", type=float, default=None,
                   help="SUMMARY の best がこの値以上に達した最初の世代を TTT として使う")
    args = p.parse_args(rest)

    if not group_specs:
        raise SystemExit("少なくとも 1 つの --group LABEL FILE... を指定してください")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    groups: list[GroupResult] = []
    for label, files in group_specs:
        print(f"=== group: {label} ({len(files)} 試行) ===")
        groups.append(aggregate(label, files, ttt_threshold=args.ttt_threshold))

    print("\n=== 図 ===")
    plot_box_summary(groups, out_dir)
    plot_fitness_curves(groups, out_dir)
    plot_block_timing(groups, out_dir)

    print("\n=== 統計検定 ===")
    comp = pairwise(groups)
    report = summarize(groups, comp)
    (out_dir / "compare_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {out_dir/'compare_report.json'}")
    print_summary(report)


if __name__ == "__main__":
    main()
