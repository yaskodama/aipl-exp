"""analyze_all.py — パイロット完了後に全分析を一括実行する。

実行内容:
  1. compare_runs.py: 3 条件比較 + post-hoc TTT@0.95
  2. compare_heatmaps.py: 条件別の building block ヒートマップ並列表示
  3. gene_transition.py: 各条件の代表試行を AST 解析 (突破点 + overlay)
  4. テキストサマリを figs/N{N}/SUMMARY.md に出力

使い方:
  python analyze_all.py --n 10           # 各条件 N=10 想定（自動的に存在するファイルを使う）
  python analyze_all.py --n 3 --tag n3   # 既存の n=3 でテスト
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
PY = sys.executable


def find_runs(label: str) -> list[Path]:
    return sorted(RUNS.glob(f"{label}_s*.ga"))


def run_cmd(cmd: list[str], desc: str) -> str:
    print(f"\n=== {desc} ===")
    print(" ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ! exit={r.returncode}")
        print("  STDERR:", r.stderr[:500])
    # 主要なログ行だけ抜粋
    for line in r.stdout.splitlines():
        if any(k in line for k in ("→", "best=", "p=", "med_", "U=", "δ=",
                                    "Cliff", "Mann", "条件", "---")):
            print("  ", line)
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="想定試行数（実在ファイル優先）")
    ap.add_argument("--tag", default=None, help="出力ディレクトリの接尾辞（既定: N{n}）")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--jump-threshold", type=float, default=0.05)
    args = ap.parse_args()
    tag = args.tag or f"N{args.n}"
    out_dir = ROOT / "figs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = ["cold", "warm", "warmstart"]
    files_by_label: dict[str, list[Path]] = {}
    for lbl in labels:
        files_by_label[lbl] = find_runs(lbl)
        print(f"[{lbl}] 検出: {len(files_by_label[lbl])} 試行")
        if not files_by_label[lbl]:
            raise SystemExit(f"  ! {lbl}_s*.ga が見つかりません")

    # ---- 1. compare_runs ----
    group_args: list[str] = []
    for lbl in labels:
        group_args += ["--group", lbl] + [str(p) for p in files_by_label[lbl]]
    run_cmd(
        [PY, "compare_runs.py", *group_args,
         "--out", str(out_dir / "compare"),
         "--ttt-threshold", str(args.threshold)],
        "条件間比較 + 統計検定 (compare_runs.py)",
    )

    # ---- 2. compare_heatmaps ----
    hm_args: list[str] = []
    for lbl in labels:
        hm_args += ["--label", lbl] + [str(p) for p in files_by_label[lbl]]
    run_cmd(
        [PY, "compare_heatmaps.py", *hm_args,
         "--out", str(out_dir / "heatmap_3way.png")],
        "条件別 building block ヒートマップ (compare_heatmaps.py)",
    )

    # ---- 3. gene_transition (各条件の最良 best fitness の試行) ----
    for lbl in labels:
        # JSON から best を見て代表試行を選ぶ
        best_run = None
        best_score = -1.0
        for ga in files_by_label[lbl]:
            from llm_agents import _level_of  # 未使用だがインポート保険
            # 雑に末尾 META から best_fitness を読む
            text = ga.read_text(encoding="utf-8")
            for line in text.splitlines():
                if line.startswith("META best_fitness="):
                    try:
                        v = float(line.split("=", 1)[1].strip().strip('"'))
                        if v > best_score:
                            best_score = v; best_run = ga
                    except ValueError:
                        pass
        if best_run is None:
            best_run = files_by_label[lbl][0]
        run_cmd(
            [PY, "gene_transition.py", str(best_run),
             "--out", str(out_dir / f"gt_{lbl}"),
             "--parser", "ast",
             "--jump-threshold", str(args.jump_threshold)],
            f"遺伝子解析 [{lbl}] best={best_score:.3f} ({best_run.name})",
        )

    # ---- 4. テキストサマリ ----
    cmp_json = out_dir / "compare" / "compare_report.json"
    if cmp_json.exists():
        rep = json.loads(cmp_json.read_text())
        md = [f"# Pilot 解析サマリ (tag={tag})\n"]
        md.append("## 条件別\n")
        md.append("| 条件 | n | best 中央値 | best SD | TTT 中央値 | 到達率 |")
        md.append("|---|---|---|---|---|---|")
        for g in rep["groups"]:
            bf = g["best_fitness"]
            ttt = g["ttt"]
            n_reached = ttt["n_reached"]
            ttt_med = ttt["median"] if ttt["median"] is not None else "—"
            md.append(f"| {g['label']} | {g['n_runs']} | {bf['median']:.3f} | "
                      f"{bf['std']:.3f} | {ttt_med} | {n_reached}/{g['n_runs']} |")
        md.append("\n## 条件間比較\n")
        md.append("| ペア | 指標 | med_a | med_b | U | p | Cliff's δ |")
        md.append("|---|---|---|---|---|---|---|")
        for key, blk in rep["comparisons"].items():
            for metric, vals in blk.items():
                md.append(f"| {key} | {metric} | {vals['median_a']:.3f} | "
                          f"{vals['median_b']:.3f} | {vals['U']:.1f} | "
                          f"{vals['p']:.4f} | {vals['cliffs_delta']:+.3f} |")
        sumpath = out_dir / "SUMMARY.md"
        sumpath.write_text("\n".join(md), encoding="utf-8")
        print(f"\n→ {sumpath}")

    print(f"\n[OK] 全成果物は {out_dir} に出力されました")


if __name__ == "__main__":
    main()
