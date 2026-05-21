"""実験 A: T3 Pomodoro 転移実験。

Todo で得た AIPL 知見 (multi_warm_s*.ga + exemplars/) を Pomodoro 題材に注入し、
問題間転移が有効か検証する。

条件:
  pomo_multi      : 7 ベンダー混成プール、履歴/exemplar なし (ベースライン)
  pomo_warm_todo  : 同上 + Todo の multi_warm_s*.ga history + Todo exemplars 注入
                    (異なる問題から得た知見の転移)
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

from llm_agents import AgentPool, multi_vendor_pool
from warm_start import wrap_with_history, wrap_with_exemplars, HistoryFedAgent, exemplar_code_fewshot, extract_history_from_runs
from ga_runner import run

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs_real"

TODO_HISTORY = sorted(RUNS_DIR.glob("multi_warm_s*.ga"))
TODO_EXEMPLARS = [
    ROOT / "exemplars/todo_aipl_minified_1line.aice",
    ROOT / "exemplars/todo_aipl_balanced_37lines.aice",  # 1行版は極端なので、可読版も
]


def pomo_multi_pool() -> AgentPool:
    """multi-vendor 7 agents, no history."""
    return multi_vendor_pool(policy="round_robin")


def pomo_warm_todo_pool() -> AgentPool:
    """multi-vendor + Todo の history (.ga) + Todo exemplar (.aice) をハイブリッド注入。

    各 agent を 2 段階でラップ:
        agent → exemplar 注入 → history 注入
    """
    base = multi_vendor_pool(policy="round_robin")
    # 履歴 fewshot
    hist = extract_history_from_runs(TODO_HISTORY, top_k=5) if TODO_HISTORY else ""
    # exemplar (実コード)
    exempl = exemplar_code_fewshot(
        TODO_EXEMPLARS,
        intro="# Todo アプリ題材で AIPL が発見した参考実装 (異なる題材ですが書きぶりが参考になる)",
    )
    combined = ""
    if hist:    combined += hist + "\n\n"
    if exempl: combined += exempl
    return AgentPool(
        [HistoryFedAgent(a, combined, name_suffix="_xfer") for a in base.agents],
        policy="round_robin",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None, help="pomo_multi/pomo_warm_todo のいずれかに絞る")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--pop", type=int, default=4)
    p.add_argument("--gens", type=int, default=3)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    conditions = [
        ("pomo_multi",     pomo_multi_pool),
        ("pomo_warm_todo", pomo_warm_todo_pool),
    ]
    if args.only:
        conditions = [(l, f) for l, f in conditions if l == args.only]

    if TODO_HISTORY:
        print(f"Todo history (.ga): {[p.name for p in TODO_HISTORY]}")
    print(f"Todo exemplars: {[p.name for p in TODO_EXEMPLARS]}")
    print()

    summary = []
    total_cost = 0.0
    for label, factory in conditions:
        for seed in seeds:
            t0 = time.time()
            out = RUNS_DIR / f"{label}_s{seed}.ga"
            gdir = RUNS_DIR / f"{label}_s{seed}_genomes"
            print(f"[{label}] seed={seed} pop={args.pop} gens={args.gens} ...", flush=True)
            r = run(
                app="pomodoro",
                pop_size=args.pop,
                gens=args.gens,
                seed=seed,
                out_path=out,
                genome_dir=gdir,
                pool=factory(),
                target_fitness=0.85,
                fitness_mode="quality",
            )
            elapsed = time.time() - t0
            print(f"  → {out.name}  best={r['best_fitness']:.3f}  "
                  f"cost=${r['cost']:.4f}  TTT={r['target_reached_gen']}  "
                  f"elapsed={elapsed:.1f}s\n", flush=True)
            summary.append((label, seed, r["best_fitness"],
                            r["target_reached_gen"], r["cost"], elapsed))
            total_cost += r["cost"]

    print("\n=== Pomodoro 転移実験サマリ ===")
    print(f"{'label':<16} {'seed':<5} {'best':<7} {'TTT':<5} {'cost':<8} {'elapsed':<8}")
    for label, seed, best, ttt, cost, elapsed in summary:
        ttt_s = "—" if ttt is None else str(ttt)
        print(f"{label:<16} {seed:<5} {best:<7.3f} {ttt_s:<5} ${cost:<7.4f} {elapsed:<8.1f}")
    print(f"\n累計コスト: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
