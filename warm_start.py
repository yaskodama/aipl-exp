"""warm_start.py — 過去試行から成功例を抽出し、LLM プロンプトに注入する。

実 Claude API パスでの ``warm-start = 履歴フィード'' を実装する。
任意の BaseAgent をラップして HistoryFedAgent を作る設計。

使い方:
    from warm_start import HistoryFedAgent, extract_history_from_runs
    from pathlib import Path

    # 過去試行 .ga から成功例を抜き出して fewshot 文字列を作る
    history = extract_history_from_runs(
        [Path("runs/warm_s101.ga"), Path("runs/warm_s102.ga")],
        top_k=3,
    )

    # 任意のエージェントをラップ
    from llm_agents import ClaudeAgent
    base = ClaudeAgent("opus", model="claude-opus-4-7")
    warm = HistoryFedAgent(base, history)

    # warm.call(prompt, op) は内部で base.call(history + prompt, op) を呼ぶ
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import gene_transition as gt
from llm_agents import BaseAgent, LLMResult


# ============================================================
# 履歴抽出
# ============================================================
@dataclass
class HistoryExample:
    """1 つの成功変異（突破点）の記録"""
    op: str
    delta_fitness: float
    parent_fitness: float
    child_fitness: float
    new_funcs: list[str]
    new_testids: list[str]


def extract_history_from_runs(
    ga_paths: list[Path],
    top_k: int = 5,
    min_delta: float = 0.05,
) -> str:
    """複数 .ga から $\\Delta f \\geq$ min_delta の成功変異を抽出し、fewshot 文字列にする。

    上位 top_k を「op が何だったか + 何が新たに獲得されたか + Δf」の形で列挙。
    実装の HTML 本文は含めず（プロンプトを肥大化させない）、機能名と op を残す。
    """
    events: list[HistoryExample] = []
    for ga in ga_paths:
        try:
            run = gt.parse_ga(ga)
        except Exception:
            continue
        stats = gt.analyze(run)
        jumps = gt.find_fitness_jumps(run, stats, delta_threshold=min_delta)
        by_id = {i.id: i for i in run.indivs}
        for j in jumps:
            child = by_id.get(j["child"])
            if not child or not j["parents"]:
                continue
            parent_fits = [by_id[p].fitness for p in j["parents"] if p in by_id]
            if not parent_fits:
                continue
            events.append(HistoryExample(
                op=j["op"],
                delta_fitness=j["delta_fitness"],
                parent_fitness=max(parent_fits),
                child_fitness=child.fitness,
                new_funcs=j["added_funcs"],
                new_testids=j["added_testids"],
            ))
    # Δf 降順、(op, new_funcs) の重複を弾く
    events.sort(key=lambda e: -e.delta_fitness)
    seen: set = set()
    uniq: list[HistoryExample] = []
    for e in events:
        sig = (e.op, tuple(e.new_funcs), tuple(e.new_testids))
        if sig in seen: continue
        seen.add(sig)
        uniq.append(e)
        if len(uniq) >= top_k: break

    if not uniq:
        return ""

    lines = ["# 過去試行で観察された fitness 向上のヒント（参考）", ""]
    lines.append("以下は、本タスクと同じ題材の過去試行で fitness を有意に上げた変更例です。")
    lines.append("同じ機能・関数を含めることで、目標到達を早められる可能性があります。")
    lines.append("")
    for i, e in enumerate(uniq, 1):
        lines.append(f"## 成功例 {i}: {e.op} で $\\Delta f$ = +{e.delta_fitness:.3f}")
        lines.append(f"   親 fitness {e.parent_fitness:.3f} → 子 fitness {e.child_fitness:.3f}")
        if e.new_funcs:
            lines.append(f"   - 新規追加関数: {', '.join(e.new_funcs)}")
        if e.new_testids:
            lines.append(f"   - 新規 data-testid: {', '.join(e.new_testids)}")
        lines.append("")
    lines.append("これらは強制ではなく参考情報です。仕様に合致する範囲で活用してください。")
    return "\n".join(lines)


# ============================================================
# History-Fed エージェント
# ============================================================
class HistoryFedAgent(BaseAgent):
    """任意の BaseAgent をラップして、init/mutation 時に履歴を注入する。

    実 LLM (ClaudeAgent) を想定: プロンプトに過去成功例が含まれることで、
    LLM が building block の獲得を早めると期待される。

    DummyAgent をラップしても history はプロンプトに含まれるだけで効果は出ない
    （dummy はプロンプトを読まない）。dummy 用の warm-start シミュレーションは
    HistoryBoostedDummy を直接使うこと。
    """

    INJECT_OPS = ("init", "mutation")

    def __init__(self, wrapped: BaseAgent, history_text: str,
                 name_suffix: str = "_warm"):
        if not isinstance(wrapped, BaseAgent):
            raise TypeError("wrapped は BaseAgent サブクラスである必要があります")
        self.wrapped = wrapped
        self.history_text = history_text or ""
        self.name = getattr(wrapped, "name", "wrapped") + name_suffix

    def call(self, prompt: str, op: str) -> LLMResult:
        if self.history_text and op in self.INJECT_OPS:
            new_prompt = self.history_text + "\n\n---\n\n" + prompt
        else:
            new_prompt = prompt
        result = self.wrapped.call(new_prompt, op)
        # name を上書き
        result.agent = self.name
        return result


# ============================================================
# 便利関数: 過去ファイルから直接ラップ
# ============================================================
def wrap_with_history(
    agent: BaseAgent,
    history_runs: list[Path],
    top_k: int = 5,
) -> HistoryFedAgent:
    """過去試行 .ga ファイル群から history_text を作って agent をラップする。"""
    history = extract_history_from_runs(history_runs, top_k=top_k)
    return HistoryFedAgent(agent, history)


# ============================================================
# Exemplar code injection — 実コードをそのまま fewshot に注入
# ============================================================
def exemplar_code_fewshot(
    exemplar_paths: list[Path],
    max_chars_per: int = 3500,
    intro: str | None = None,
) -> str:
    """`.aice` (HTML) の実コードを fewshot 文字列にして返す。

    metadata 注入 (extract_history_from_runs) と違い、**実装そのものを** LLM に提示する。
    短く高品質な exemplar (例: 1-LOC champion) を seed として使うのに有効。
    """
    if not exemplar_paths:
        return ""
    lines: list[str] = []
    lines.append(intro or "# 過去のパイロットで発見された高品質実装の例")
    lines.append("")
    lines.append("以下は同じ仕様を満たす高品質な実装の例です。書きぶり・短さ・関数構成を参考にしてください。")
    lines.append("")
    for i, path in enumerate(exemplar_paths, 1):
        code = path.read_text(encoding="utf-8")
        truncated = ""
        if len(code) > max_chars_per:
            code = code[:max_chars_per]
            truncated = " ...(省略)"
        lines.append(f"## 例 {i} ({path.stem}, {len(code)} chars):")
        lines.append("```html")
        lines.append(code + truncated)
        lines.append("```")
        lines.append("")
    lines.append("---")
    lines.append("これらの実装スタイルを参考に、新しい実装を作成してください。")
    return "\n".join(lines)


def wrap_with_exemplars(
    agent: BaseAgent,
    exemplar_paths: list[Path],
    max_chars_per: int = 3500,
) -> HistoryFedAgent:
    """高品質 exemplar のコード本体を fewshot として注入してエージェントをラップ。"""
    fewshot = exemplar_code_fewshot(exemplar_paths, max_chars_per=max_chars_per)
    return HistoryFedAgent(agent, fewshot, name_suffix="_exempl")


# ============================================================
# 動作確認 (smoke test)
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python warm_start.py <ga_file> [<ga_file> ...]")
        sys.exit(1)
    paths = [Path(p) for p in sys.argv[1:]]
    text = extract_history_from_runs(paths, top_k=5)
    print(text)
