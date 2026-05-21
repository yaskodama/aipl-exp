"""compare_heatmaps.py — 複数試行の関数 building block ヒートマップを横並び比較する。

使い方:
    python compare_heatmaps.py \
        --label warm      runs/warm_s101.ga \
        --label warmstart runs/warmstart_s101.ga \
        --out figs/heatmap_compare.png

論文の "building block の獲得が warm-start で早まっている" 主張を示す図を作る。
"""

from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = [
    "Hiragino Sans", "Hiragino Maru Gothic Pro", "Yu Gothic",
    "Meiryo", "Noto Sans CJK JP", "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False

import gene_transition as gt


def collect(ga_paths: list[Path]) -> tuple[dict, dict]:
    """複数 .ga を平均化したヒートマップ用データを作る。"""
    gen_size_sum: Counter = Counter()
    gen_func_pop_sum: dict[int, Counter] = {}
    for ga in ga_paths:
        run = gt.parse_ga(ga)
        s = gt.analyze(run)
        for g, n in s.gen_size.items():
            gen_size_sum[g] += n
            if g not in gen_func_pop_sum:
                gen_func_pop_sum[g] = Counter()
            for name, count in s.gen_func_pop[g].items():
                gen_func_pop_sum[g][name] += count
    return gen_func_pop_sum, dict(gen_size_sum)


def common_order(groups: dict) -> list[str]:
    """全条件で初出が早いものを上に並べる順序を決める。"""
    first_app: dict[str, int] = {}
    for label, (pop, size) in groups.items():
        for g in sorted(size):
            for name, count in pop.get(g, {}).items():
                if count > 0 and (name not in first_app or g < first_app[name]):
                    first_app[name] = g
    return sorted(first_app, key=lambda n: (first_app[n], n))


def parse_groups(argv: list[str]):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--label":
            label = argv[i + 1]
            j = i + 2
            files = []
            while j < len(argv) and not argv[j].startswith("--"):
                files.append(Path(argv[j])); j += 1
            out.append((label, files))
            i = j
        else:
            i += 1
    return out


def main():
    raw = sys.argv[1:]
    group_specs = parse_groups(raw)
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--top-k", type=int, default=12)
    args, _ = p.parse_known_args(raw)

    if not group_specs:
        raise SystemExit("--label LABEL FILE... を 1 つ以上指定してください")

    groups: dict = {}
    for label, files in group_specs:
        groups[label] = collect(files)
        print(f"[{label}] {len(files)} 試行を集約")

    order = common_order(groups)[: args.top_k]
    print(f"上位 {len(order)} 関数: {order}")

    fig, axes = plt.subplots(1, len(groups), figsize=(max(9, 4.2 * len(groups)), 5.2))
    if len(groups) == 1:
        axes = [axes]
    for idx, (ax, (label, (pop, size))) in enumerate(zip(axes, groups.items())):
        gens = sorted(size.keys())
        mat = np.zeros((len(order), len(gens)))
        for i, name in enumerate(order):
            for j, g in enumerate(gens):
                mat[i, j] = pop.get(g, Counter()).get(name, 0) / max(size[g], 1)
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(gens))); ax.set_xticklabels([f"g{g}" for g in gens])
        ax.set_yticks(range(len(order)))
        if idx == 0:
            ax.set_yticklabels(order)
        else:
            ax.set_yticklabels([""] * len(order))
        ax.set_title(label, fontsize=12)
        for i in range(len(order)):
            for j in range(len(gens)):
                v = mat[i, j]
                if v > 0:
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            color="white" if v > 0.55 else "#333", fontsize=8)
    fig.suptitle("関数 building block の集団内出現率（条件別）", fontsize=13)
    fig.colorbar(im, ax=axes, label="集団内出現率", shrink=0.7)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {args.out}")


if __name__ == "__main__":
    main()
