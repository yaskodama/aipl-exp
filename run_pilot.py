"""パイロット実験ランナー: cold (低品質エージェント) vs warm (高品質エージェント)。

実 Claude は使わず、強化したDummyAgent (7 段階レベル) で進化挙動を模擬する。
ANTHROPIC_API_KEY 設定時に claude_pool() に差し替えれば本実験になる。
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

from llm_agents import AgentPool, DummyAgent, HistoryBoostedDummy
from ga_runner import run

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)


def cold_pool() -> AgentPool:
    """低品質エージェント = 単一 LLM ベースラインの代理。"""
    return AgentPool([
        DummyAgent("cold_a", quality=0.40),
        DummyAgent("cold_b", quality=0.35),
        DummyAgent("cold_c", quality=0.45),
    ], policy="round_robin")


def warm_pool() -> AgentPool:
    """高品質エージェント = AIPL マルチエージェントの代理（cold start）。"""
    return AgentPool([
        DummyAgent("warm_opus",   quality=0.85),
        DummyAgent("warm_sonnet", quality=0.75),
        DummyAgent("warm_haiku",  quality=0.65),
    ], policy="round_robin")


def warmstart_pool() -> AgentPool:
    """履歴ブートストラップ AIPL（実験 4-2: warm-start = 過去試行の知見を注入）。

    quality は warm_pool と同じだが、init が L2 から出発する点が異なる。
    """
    return AgentPool([
        HistoryBoostedDummy("ws_opus",   quality=0.85, history_floor=2),
        HistoryBoostedDummy("ws_sonnet", quality=0.75, history_floor=2),
        HistoryBoostedDummy("ws_haiku",  quality=0.65, history_floor=2),
    ], policy="round_robin")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None,
                   help="cold/warm/warmstart のいずれかに絞る")
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--seeds", default="101,102,103",
                   help="カンマ区切りの seed 値")
    p.add_argument("--pop", type=int, default=4)
    p.add_argument("--gens", type=int, default=5)
    args = p.parse_args()

    pop = args.pop
    gens = args.gens
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    summary = []

    conditions = [
        ("cold",      cold_pool),
        ("warm",      warm_pool),
        ("warmstart", warmstart_pool),
    ]
    if args.only:
        conditions = [(l, f) for l, f in conditions if l == args.only]

    for label, pool_factory in conditions:
        for seed in seeds:
            t0 = time.time()
            out = RUNS_DIR / f"{label}_s{seed}.ga"
            gdir = RUNS_DIR / f"{label}_s{seed}_genomes"
            print(f"\n[{label}] seed={seed} pop={pop} gens={gens} ...", flush=True)
            r = run(
                app="todo",
                pop_size=pop,
                gens=gens,
                seed=seed,
                out_path=out,
                genome_dir=gdir,
                pool=pool_factory(),
                target_fitness=args.target,
            )
            elapsed = time.time() - t0
            print(f"  → {out.name}  best={r['best_fitness']:.3f}  "
                  f"calls={r['calls']}  TTT={r['target_reached_gen']}  "
                  f"elapsed={elapsed:.1f}s", flush=True)
            summary.append((label, seed, r["best_fitness"],
                            r["target_reached_gen"], elapsed))

    print("\n=== pilot summary ===")
    print(f"{'label':<6} {'seed':<5} {'best':<7} {'TTT':<5} {'elapsed':<8}")
    for label, seed, best, ttt, elapsed in summary:
        ttt_s = "—" if ttt is None else str(ttt)
        print(f"{label:<6} {seed:<5} {best:<7.3f} {ttt_s:<5} {elapsed:<8.1f}")


if __name__ == "__main__":
    main()
