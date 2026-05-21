"""
GA 実験ランナー: AIPL の進化計算ループ

使い方:
    python ga_runner.py --app todo --pop 4 --gens 8 --seed 42 \
        --out runs/todo_s42.ga

各個体の HTML は --genome-dir に <id>.aice として保存される。
（.aice は logic-cir の流儀に合わせた拡張子。中身は HTML テキスト）
"""

from __future__ import annotations
import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fitness_harness import evaluate, FitnessResult
from llm_agents import AgentPool, default_pool, build_prompt, strip_codefence


# ============================================================
# データクラス
# ============================================================
@dataclass
class Individual:
    id: str
    gen: int
    op: str
    parents: list[str]
    agent: str
    call_id: str
    fitness: float
    passed: int
    total: int
    loc: int
    failed_tests: list[str]
    genome_path: str
    notes: str = ""


@dataclass
class EventRecord:
    id: str
    agent: str
    task: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost: float
    parents: list[str] = field(default_factory=list)


# ============================================================
# .ga ロガー
# ============================================================
class GALogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w", encoding="utf-8")

    def meta(self, **kw):
        for k, v in kw.items():
            self._w(f"META {k}={self._fmt(v)}")

    def gen(self, n: int):
        self._w("")
        self._w(f"GEN {n}")

    def indiv(self, ind: Individual):
        parts = [
            f"id={ind.id}",
            f"op={ind.op}",
            f"fitness={ind.fitness:.4f}",
            f"parents=[{','.join(ind.parents)}]",
            f"agent={ind.agent or '-'}",
            f"call={ind.call_id or '-'}",
            f"passed={ind.passed}/{ind.total}",
            f"loc={ind.loc}",
            f'genome_ref="{ind.genome_path}"',
        ]
        if ind.notes:
            parts.append(f'notes="{ind.notes}"')
        self._w("INDIV " + " ".join(parts))

    def event(self, ev: EventRecord):
        parts = [
            "llm_call",
            f"id={ev.id}",
            f"agent={ev.agent}",
            f"task={ev.task}",
            f"tokens_in={ev.tokens_in}",
            f"tokens_out={ev.tokens_out}",
            f"latency_ms={ev.latency_ms}",
            f"cost={ev.cost:.6f}",
        ]
        if ev.parents:
            parts.append(f"parents={','.join(ev.parents)}")
        self._w("EVENT " + " ".join(parts))

    def summary(self, gen: int, fitnesses: list[float],
                diversity: float, elapsed_ms: int, cum_cost: float):
        best, mean, worst = max(fitnesses), sum(fitnesses)/len(fitnesses), min(fitnesses)
        self._w(
            f"SUMMARY gen={gen} best={best:.4f} mean={mean:.4f} worst={worst:.4f} "
            f"diversity={diversity:.4f} elapsed_ms={elapsed_ms} "
            f"cumulative_cost={cum_cost:.6f}"
        )

    def comment(self, text: str):
        self._w(f"# {text}")

    def close(self):
        self.f.close()

    def _w(self, line: str):
        self.f.write(line + "\n")
        self.f.flush()

    @staticmethod
    def _fmt(v):
        if isinstance(v, str):
            return f'"{v}"'
        return str(v)


# ============================================================
# 多様性指標: ゲノム長の標準偏差を簡易プロキシ
# ============================================================
def diversity(genomes: list[str]) -> float:
    if len(genomes) < 2: return 0.0
    n = len(genomes)
    pairs = 0
    total = 0
    for i in range(n):
        for j in range(i+1, n):
            a, b = genomes[i], genomes[j]
            # 文字列の正規化編集距離の安価な近似: 長さ差 / max
            d = abs(len(a) - len(b)) / max(len(a), len(b), 1)
            total += d
            pairs += 1
    return round(total / max(pairs, 1), 4)


# ============================================================
# 選択
# ============================================================
def tournament_select(population: list[Individual], k: int = 3) -> Individual:
    return max(random.sample(population, k=min(k, len(population))), key=lambda x: x.fitness)


# ============================================================
# メインループ
# ============================================================
def run(app: str, pop_size: int, gens: int, seed: int,
        out_path: Path, genome_dir: Path,
        pool: Optional[AgentPool] = None,
        target_fitness: float = 0.999,
        fitness_mode: str = "test") -> dict:
    random.seed(seed)
    pool = pool or default_pool()
    log = GALogger(out_path)
    genome_dir.mkdir(parents=True, exist_ok=True)

    log.comment("GA v1 — AIPL アプリ進化実験")
    log.meta(
        problem=app, condition="AIPL-multi-agent", population=pop_size,
        generations=gens, seed=seed,
        start_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
        target_fitness=target_fitness,
        fitness_mode=fitness_mode,
    )

    call_counter = 0
    total_cost = 0.0
    population: list[Individual] = []

    def save_genome(text: str, ind_id: str) -> str:
        p = genome_dir / f"{ind_id}.aice"
        p.write_text(text, encoding="utf-8")
        return str(p.relative_to(out_path.parent.parent)) if out_path.parent.parent in p.parents else str(p)

    def make_call_id() -> str:
        nonlocal call_counter
        call_counter += 1
        return f"c{call_counter:04d}"

    # ---- Gen 0: 初期化 ----
    log.gen(0)
    t_gen0 = time.time()
    for i in range(pop_size):
        agent = pool.pick("init")
        prompt = build_prompt("init", app)
        r = agent.call(prompt, "init")
        html = strip_codefence(r.text)
        call_id = make_call_id()
        log.event(EventRecord(
            id=call_id, agent=r.agent, task="init",
            tokens_in=r.tokens_in, tokens_out=r.tokens_out,
            latency_ms=r.latency_ms, cost=r.cost,
        ))
        total_cost += r.cost

        ind_id = f"g0i{i}"
        gpath = save_genome(html, ind_id)
        fr = evaluate(html, app=app, fitness_mode=fitness_mode)
        ind = Individual(
            id=ind_id, gen=0, op="init", parents=[], agent=r.agent,
            call_id=call_id, fitness=fr.fitness, passed=fr.passed, total=fr.total,
            loc=fr.loc, failed_tests=fr.failed_tests, genome_path=gpath,
            notes=fr.error,
        )
        log.indiv(ind)
        population.append(ind)

    elapsed_ms = int((time.time() - t_gen0) * 1000)
    log.summary(0, [p.fitness for p in population],
                diversity([Path(p.genome_path).read_text() for p in population]),
                elapsed_ms, total_cost)

    target_reached_gen = None
    if max(p.fitness for p in population) >= target_fitness:
        target_reached_gen = 0

    # ---- Gen 1..N ----
    for g in range(1, gens + 1):
        log.gen(g)
        t_gen = time.time()
        next_pop: list[Individual] = []

        # エリート保存 (top 1)
        elite = max(population, key=lambda x: x.fitness)
        elite_clone = Individual(
            id=f"g{g}i0", gen=g, op="elite", parents=[elite.id],
            agent="-", call_id="-", fitness=elite.fitness,
            passed=elite.passed, total=elite.total, loc=elite.loc,
            failed_tests=elite.failed_tests, genome_path=elite.genome_path,
        )
        log.indiv(elite_clone)
        next_pop.append(elite_clone)

        # 残りを GA 演算で生成
        for i in range(1, pop_size):
            ind_id = f"g{g}i{i}"
            # 演算子選択: 序盤は crossover、改善が頭打ちなら mutation 寄り
            r_op = random.random()
            if r_op < 0.4 and len(population) >= 2:
                op = "crossover"
            elif r_op < 0.9:
                op = "mutation"
            else:
                op = "simplify"

            if op == "crossover":
                pa = tournament_select(population)
                pb = tournament_select(population)
                pa_html = Path(pa.genome_path).read_text(encoding="utf-8")
                pb_html = Path(pb.genome_path).read_text(encoding="utf-8")
                prompt = build_prompt(
                    "crossover", app,
                    parent_a=pa_html, parent_b=pb_html,
                    a_passed=[f"T{idx+1}" for idx in range(pa.passed)],
                    b_passed=[f"T{idx+1}" for idx in range(pb.passed)],
                )
                parents_ids = [pa.id, pb.id]
            else:
                pa = tournament_select(population)
                pa_html = Path(pa.genome_path).read_text(encoding="utf-8")
                if op == "mutation":
                    prompt = build_prompt("mutation", app,
                                          parent=pa_html, failed=pa.failed_tests)
                else:
                    prompt = build_prompt("simplify", app, parent=pa_html)
                parents_ids = [pa.id]

            agent = pool.pick(op)
            r = agent.call(prompt, op)
            html = strip_codefence(r.text)
            call_id = make_call_id()
            log.event(EventRecord(
                id=call_id, agent=r.agent, task=op,
                tokens_in=r.tokens_in, tokens_out=r.tokens_out,
                latency_ms=r.latency_ms, cost=r.cost,
                parents=parents_ids,
            ))
            total_cost += r.cost

            gpath = save_genome(html, ind_id)
            fr = evaluate(html, app=app, fitness_mode=fitness_mode)
            ind = Individual(
                id=ind_id, gen=g, op=op, parents=parents_ids,
                agent=r.agent, call_id=call_id,
                fitness=fr.fitness, passed=fr.passed, total=fr.total,
                loc=fr.loc, failed_tests=fr.failed_tests,
                genome_path=gpath, notes=fr.error,
            )
            log.indiv(ind)
            next_pop.append(ind)

        population = next_pop
        elapsed_ms = int((time.time() - t_gen) * 1000)
        log.summary(g, [p.fitness for p in population],
                    diversity([Path(p.genome_path).read_text() for p in population]),
                    elapsed_ms, total_cost)

        if target_reached_gen is None and max(p.fitness for p in population) >= target_fitness:
            target_reached_gen = g

    # 終了メタ
    best = max(population, key=lambda x: x.fitness)
    log.comment("試行完了")
    log.meta(
        end_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
        status="success" if target_reached_gen is not None else "stopped",
        best_id=best.id, best_fitness=f"{best.fitness:.4f}",
        total_llm_calls=call_counter,
        total_cost=f"{total_cost:.6f}",
        generations_to_target=(target_reached_gen if target_reached_gen is not None else "-"),
    )
    log.close()
    return {
        "best_fitness": best.fitness,
        "best_id": best.id,
        "calls": call_counter,
        "cost": total_cost,
        "target_reached_gen": target_reached_gen,
        "out": str(out_path),
    }


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app", default="todo", choices=["todo", "calculator"])
    p.add_argument("--pop", type=int, default=4)
    p.add_argument("--gens", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="出力 .ga ファイルパス")
    p.add_argument("--genome-dir", default=None,
                   help="個体ゲノム保存ディレクトリ（デフォルト: <out>.genomes）")
    args = p.parse_args()
    out = Path(args.out)
    gdir = Path(args.genome_dir) if args.genome_dir else out.with_suffix("").parent / (out.stem + "_genomes")
    summary = run(args.app, args.pop, args.gens, args.seed, out, gdir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
