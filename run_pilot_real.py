"""実 LLM (Claude + GPT + Gemini) で AIPL パイロットを実行する (A' プラン)。

3 条件:
  cold        : Haiku 4.5 のみ (単一 LLM ベースライン)
  multi       : 7 エージェント混成プール (マルチベンダー AIPL)
  multi_warm  : 同上 + 過去成功例の fewshot 注入

評価モード: quality (pass_rate × (0.3 + 0.7*LOC_score)) - コード品質方向に駆動。

使い方:
    python run_pilot_real.py --seeds 1,2,3
    python run_pilot_real.py --only cold --seeds 1
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

from llm_agents import AgentPool, ClaudeAgent, OpenAIAgent, GeminiAgent, multi_vendor_pool
from warm_start import wrap_with_history
from ga_runner import run

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs_real"
RUNS_DIR.mkdir(exist_ok=True)


def cold_pool() -> AgentPool:
    """単一 LLM ベースライン: Haiku 4.5 のみ。"""
    return AgentPool([
        ClaudeAgent("haiku", model="claude-haiku-4-5"),
    ], policy="round_robin")


def warm_pool() -> AgentPool:
    """マルチベンダー混成 (cold-start = 履歴なし)。"""
    return multi_vendor_pool(policy="round_robin")


def warmstart_pool(history_paths: list[Path]) -> AgentPool:
    """マルチベンダー混成 + 過去成功例の fewshot 注入 (warm-start)。"""
    base = multi_vendor_pool(policy="round_robin")
    return AgentPool(
        [wrap_with_history(a, history_paths, top_k=5) for a in base.agents],
        policy="round_robin",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None,
                   help="cold/multi/multi_warm のいずれかに絞る")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--pop", type=int, default=4)
    p.add_argument("--gens", type=int, default=3,
                   help="real LLM はコストが高いため既定 3 世代")
    p.add_argument("--mode", default="quality", choices=["test", "quality"])
    p.add_argument("--target", type=float, default=0.85,
                   help="quality mode の到達閾値 (test mode なら 0.95 等)")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # history は warm 試行の .ga から抽出する想定（warm を先に走らせる）
    # --only multi_warm 単独実行時は、ディスク上の既存 multi_*.ga を自動検出して使う
    warm_results: list[Path] = sorted(RUNS_DIR.glob("multi_s*.ga"))
    if warm_results:
        print(f"既存の multi 試行を warm-start の history として利用: {[p.name for p in warm_results]}")

    conditions = [
        ("cold",       lambda: cold_pool()),
        ("multi",      lambda: warm_pool()),
        ("multi_warm", lambda: warmstart_pool(warm_results) if warm_results else warm_pool()),
    ]
    if args.only:
        conditions = [(l, f) for l, f in conditions if l == args.only]

    summary = []
    total_cost = 0.0
    for label, factory in conditions:
        for seed in seeds:
            t0 = time.time()
            out = RUNS_DIR / f"{label}_s{seed}.ga"
            gdir = RUNS_DIR / f"{label}_s{seed}_genomes"
            print(f"\n[{label}] seed={seed} pop={args.pop} gens={args.gens} mode={args.mode} ...", flush=True)
            r = run(
                app="todo",
                pop_size=args.pop,
                gens=args.gens,
                seed=seed,
                out_path=out,
                genome_dir=gdir,
                pool=factory(),
                target_fitness=args.target,
                fitness_mode=args.mode,
            )
            elapsed = time.time() - t0
            print(f"  → {out.name}  best={r['best_fitness']:.3f}  "
                  f"calls={r['calls']}  cost=${r['cost']:.4f}  "
                  f"TTT={r['target_reached_gen']}  elapsed={elapsed:.1f}s",
                  flush=True)
            summary.append((label, seed, r["best_fitness"],
                            r["target_reached_gen"], r["cost"], elapsed))
            total_cost += r["cost"]

            # warm の出力を warmstart の history source にする
            if label == "multi":
                warm_results.append(out)

    print("\n=== real pilot summary ===")
    print(f"{'label':<11} {'seed':<5} {'best':<7} {'TTT':<5} {'cost':<8} {'elapsed':<8}")
    for label, seed, best, ttt, cost, elapsed in summary:
        ttt_s = "—" if ttt is None else str(ttt)
        print(f"{label:<11} {seed:<5} {best:<7.3f} {ttt_s:<5} ${cost:<7.4f} {elapsed:<8.1f}")
    print(f"\n累計 LLM コスト: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
