"""実験 C: 1-LOC exemplar を fewshot 注入することで、Haiku 単体が AIPL レベルの
圧縮コードを書けるようになるか検証する。

比較:
  cold (既存 n=3)           : Haiku のみ、履歴/exemplar なし
  haiku_exempl (新規 n=3)   : Haiku のみ + 1-LOC + 18-LOC exemplar の fewshot 注入

仮説: warm-start (exemplar code 注入) は単一の弱モデルでも AIPL レベルの圧縮を実現する。
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

from llm_agents import AgentPool, ClaudeAgent
from warm_start import wrap_with_exemplars
from ga_runner import run

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs_real"
EXEMPLARS = [
    ROOT / "exemplars/todo_aipl_minified_1line.aice",
    ROOT / "exemplars/todo_aipl_compact_18lines.aice",
]


def haiku_exempl_pool() -> AgentPool:
    """Haiku + 1-LOC/18-LOC exemplar code を fewshot 注入。"""
    base = ClaudeAgent("haiku", model="claude-haiku-4-5")
    wrapped = wrap_with_exemplars(base, EXEMPLARS, max_chars_per=3500)
    return AgentPool([wrapped], policy="round_robin")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--pop", type=int, default=4)
    p.add_argument("--gens", type=int, default=3)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    summary = []
    total_cost = 0.0

    print(f"exemplars: {[p.name for p in EXEMPLARS]}")
    for seed in seeds:
        t0 = time.time()
        out = RUNS_DIR / f"haiku_exempl_s{seed}.ga"
        gdir = RUNS_DIR / f"haiku_exempl_s{seed}_genomes"
        print(f"\n[haiku_exempl] seed={seed} pop={args.pop} gens={args.gens} ...", flush=True)
        r = run(
            app="todo",
            pop_size=args.pop,
            gens=args.gens,
            seed=seed,
            out_path=out,
            genome_dir=gdir,
            pool=haiku_exempl_pool(),
            target_fitness=0.85,
            fitness_mode="quality",
        )
        elapsed = time.time() - t0
        print(f"  → {out.name}  best={r['best_fitness']:.3f}  "
              f"cost=${r['cost']:.4f}  TTT={r['target_reached_gen']}  "
              f"elapsed={elapsed:.1f}s", flush=True)
        summary.append((seed, r["best_fitness"], r["target_reached_gen"], r["cost"], elapsed))
        total_cost += r["cost"]

    print("\n=== haiku_exempl summary ===")
    print(f"{'seed':<5} {'best':<7} {'TTT':<5} {'cost':<8} {'elapsed':<8}")
    for seed, best, ttt, cost, elapsed in summary:
        ttt_s = "—" if ttt is None else str(ttt)
        print(f"{seed:<5} {best:<7.3f} {ttt_s:<5} ${cost:<7.4f} {elapsed:<8.1f}")
    print(f"\n累計 LLM コスト: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
