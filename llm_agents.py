"""
LLM エージェント層: AIPL の "マルチエージェント" 部分

3 種類のエージェントを提供:
  - DummyAgent  : API キー不要、ハードコード変種を返す（GA ループの動作検証用）
  - ClaudeAgent : Anthropic SDK 経由の実 Claude（Opus 4.7 / Sonnet 4.6 / Haiku 4.5）
  - RealAgent   : 他プロバイダ用の抽象テンプレート（OpenAI / Gemini を入れたい人向け）

ANTHROPIC_API_KEY が環境変数に設定されていれば default_pool() が
自動的に claude_pool() を返します。
"""

from __future__ import annotations
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import anthropic  # type: ignore
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    from openai import OpenAI  # type: ignore
    import openai  # type: ignore
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

try:
    from google import genai  # type: ignore
    from google.genai import types as _genai_types  # type: ignore
    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False


def _load_dotenv() -> None:
    """同じディレクトリの .env を読んで os.environ に流し込む（最小実装）。
    既存の環境変数は上書きしない。"""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()

SPEC_DIR = Path(__file__).parent / "apps"


# ============================================================
# プロンプトテンプレート
# ============================================================
SYSTEM_PROMPT = """\
あなたは単一 HTML ファイルでアプリを実装する熟練ソフトウェアエンジニアです。
出力は HTML 1 ファイルのみ。マークダウンコードフェンスや解説は一切付けないこと。
先頭は必ず <!DOCTYPE html> から始める。
"""

INIT_TEMPLATE = """\
以下の仕様に従って、単一 HTML ファイル（CSS, JS をインライン）でアプリを実装してください。

【仕様】
{spec}

要求:
- 1 ファイル完結、外部ライブラリ・CDN 不可
- 全ての data-testid を仕様通り付与すること
- 出力は HTML のみ
"""

MUTATION_TEMPLATE = """\
以下のアプリ実装には合格しないテストがあります。最小修正で改善してください。

【仕様】
{spec}

【失敗したテスト】
{failed}

【現在の実装】
{parent}

要求:
- 出力は修正後の単一 HTML ファイルのみ
- 過剰な書き換えはせず、失敗テストを通すための最小修正に留める
"""

CROSSOVER_TEMPLATE = """\
以下の 2 つの実装は別々のテストで強みを持ちます。両者の良いところを統合した実装を作ってください。

【仕様】
{spec}

【親 A の合格テスト: {a_passed}】
{parent_a}

【親 B の合格テスト: {b_passed}】
{parent_b}

要求:
- 各機能の出典が分かるよう、対応する箇所に // from parent A または // from parent B のコメントを残すこと
- 出力は HTML のみ
"""

SIMPLIFY_TEMPLATE = """\
以下のアプリ実装について、振る舞い（テスト合格状況）を変えずに行数を削減してください。

【仕様】
{spec}

【現在の実装】
{parent}

要求:
- 機能を削除してはならない
- 出力は HTML のみ
"""


# ============================================================
# 共通 API
# ============================================================
@dataclass
class LLMResult:
    text: str
    agent: str
    op: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    cost: float = 0.0


def load_spec(app: str) -> str:
    p = SPEC_DIR / app / "spec.txt"
    return p.read_text(encoding="utf-8")


def build_prompt(op: str, app: str, **kw) -> str:
    spec = load_spec(app)
    if op == "init":
        return INIT_TEMPLATE.format(spec=spec)
    if op == "mutation":
        return MUTATION_TEMPLATE.format(spec=spec, failed="\n".join(kw["failed"]), parent=kw["parent"])
    if op == "crossover":
        return CROSSOVER_TEMPLATE.format(
            spec=spec,
            a_passed=", ".join(kw.get("a_passed", [])),
            b_passed=", ".join(kw.get("b_passed", [])),
            parent_a=kw["parent_a"], parent_b=kw["parent_b"],
        )
    if op == "simplify":
        return SIMPLIFY_TEMPLATE.format(spec=spec, parent=kw["parent"])
    raise ValueError(f"unknown op: {op}")


def strip_codefence(text: str) -> str:
    """LLM がたまに ```html ... ``` で囲うので取り除く"""
    m = re.search(r"```(?:html)?\s*\n?([\s\S]*?)```", text)
    return m.group(1).strip() if m else text.strip()


# ============================================================
# ダミーエージェント（実 LLM なしで GA を回すためのスタブ）
# ============================================================
# Todo アプリの 7 段階（L0 から L6）を生成する。各レベルが新しい機能と
# それに対応する関数 / data-testid を追加する設計なので、gene_transition.py
# の building block 解析で機能獲得の系列が綺麗に観察できる。
#
#   L0: 骨組みのみ（テスト T1 のみ通る）
#   L1: + addTodo, Enter/button での追加（T2, T3, T4 が通る）
#   L2: + deleteTodo（T5 が通る）
#   L3: + toggleTodo + 完了スタイル（T6, T7 が通る）
#   L4: + filter（T8, T9, T10 が通る）
#   L5: + localStorage 永続化（T11 が通る）
#   L6: + clearCompleted（T12 が通る、満点）

_LEVEL_RE = re.compile(r"<!--\s*LVL=(\d+)\s*-->")


def _level_of(html: str) -> int:
    m = _LEVEL_RE.search(html or "")
    return int(m.group(1)) if m else 0


def _todo_html(level: int) -> str:
    """指定レベルの Todo アプリ HTML を生成する。"""
    level = max(0, min(6, level))
    style = ".done{text-decoration:line-through;}" if level >= 3 else ""
    # filter ボタン（L4+ では data-f 属性で機能、それ未満は飾り）
    if level >= 4:
        filter_html = ('<button data-testid="filter-all" data-f="all">All</button>'
                       '<button data-testid="filter-active" data-f="active">Active</button>'
                       '<button data-testid="filter-completed" data-f="completed">Done</button>')
    else:
        filter_html = ('<button data-testid="filter-all"></button>'
                       '<button data-testid="filter-active"></button>'
                       '<button data-testid="filter-completed"></button>')
    clear_html = '<button data-testid="clear-completed">Clear Done</button>'

    js = [f'<!-- LVL={level} -->']
    # storage helpers
    if level >= 5:
        js.append("function loadTodos(){try{return JSON.parse(localStorage.getItem('todos')||'[]')}catch(e){return []}}")
        js.append("function persistTodos(items){localStorage.setItem('todos', JSON.stringify(items))}")
    else:
        js.append("function loadTodos(){return []}")
        js.append("function persistTodos(){}")
    js.append("let todos = loadTodos();")
    js.append("let currentFilter = 'all';")
    if level >= 1:
        js.append("function addTodo(text){if(!text)return;todos.push({text,done:false});persistTodos(todos);render();}")
    if level >= 2:
        js.append("function deleteTodo(i){todos.splice(i,1);persistTodos(todos);render();}")
    if level >= 3:
        js.append("function toggleTodo(i){todos[i].done=!todos[i].done;persistTodos(todos);render();}")
    if level >= 4:
        js.append("function setFilter(f){currentFilter=f;render();}")
    if level >= 6:
        js.append("function clearCompleted(){todos=todos.filter(t=>!t.done);persistTodos(todos);render();}")
    # render
    render = [
        "function render(){",
        "  const list=document.querySelector('[data-testid=\"todo-list\"]');",
        "  list.innerHTML='';",
        "  todos.forEach((t,i)=>{",
        "    if(currentFilter==='active'&&t.done)return;",
        "    if(currentFilter==='completed'&&!t.done)return;",
        "    const li=document.createElement('li');li.dataset.testid='todo-item';",
        "    const span=document.createElement('span');span.dataset.testid='todo-text';span.textContent=t.text;",
        "    if(t.done)span.classList.add('done');",
        "    const tg=document.createElement('input');tg.type='checkbox';tg.dataset.testid='todo-toggle';tg.checked=t.done;",
    ]
    if level >= 3:
        render.append("    tg.addEventListener('change',()=>toggleTodo(i));")
    render += [
        "    const del=document.createElement('button');del.dataset.testid='todo-delete';del.textContent='x';",
    ]
    if level >= 2:
        render.append("    del.addEventListener('click',()=>deleteTodo(i));")
    render += [
        "    li.appendChild(span);li.appendChild(tg);li.appendChild(del);",
        "    list.appendChild(li);",
        "  });",
        "}",
    ]
    js.extend(render)
    # bootstrap
    boot = ["document.addEventListener('DOMContentLoaded',()=>{"]
    if level >= 1:
        boot += [
            "  const inp=document.querySelector('[data-testid=\"new-todo-input\"]');",
            "  const btn=document.querySelector('[data-testid=\"add-btn\"]');",
            "  inp.addEventListener('keydown',e=>{if(e.key==='Enter'){addTodo(inp.value);inp.value='';}});",
            "  btn.addEventListener('click',()=>{addTodo(inp.value);inp.value='';});",
        ]
    if level >= 4:
        boot.append("  document.querySelectorAll('[data-f]').forEach(b=>b.addEventListener('click',()=>setFilter(b.dataset.f)));")
    if level >= 6:
        boot.append("  document.querySelector('[data-testid=\"clear-completed\"]').addEventListener('click',clearCompleted);")
    boot.append("  render();")
    boot.append("});")
    js.extend(boot)
    js_src = "\n".join(js)

    return (f'<!DOCTYPE html><html><head><meta charset="UTF-8"><style>{style}</style></head><body>'
            f'<input data-testid="new-todo-input"/>'
            f'<button data-testid="add-btn">Add</button>'
            f'<ul data-testid="todo-list"></ul>'
            f'{filter_html}{clear_html}'
            f'<script>{js_src}</script></body></html>')


class BaseAgent:
    name = "base"
    def call(self, prompt: str, op: str) -> LLMResult: ...


class HistoryBoostedDummy(BaseAgent):
    """warm-start ダミー: 過去試行の成功例で初期レベルを引き上げる。

    実 LLM での実験 4-2（履歴フィード）の挙動を、init の出発レベルを
    history_floor 以上にすることで模擬する。mutation/crossover は通常通り。

    例: history_floor=2 は ``過去試行が deleteTodo まで到達済み'' を意味し、
    新試行の init は L2 から出発する。
    """
    def __init__(self, name: str = "warm_dummy", quality: float = 0.8,
                 history_floor: int = 2):
        self.name = name
        self.quality = quality
        self.history_floor = max(0, min(6, history_floor))

    def _parent_levels(self, prompt: str) -> list[int]:
        return [int(m.group(1)) for m in _LEVEL_RE.finditer(prompt)]

    def call(self, prompt: str, op: str) -> LLMResult:
        t0 = time.time()
        time.sleep(0.01)
        parents = self._parent_levels(prompt)
        if op == "init":
            # 履歴ブートストラップ: history_floor から始まり、品質に応じて +1
            level = self.history_floor + (1 if random.random() < self.quality else 0)
            level = min(level, 6)
        elif op == "mutation":
            base = parents[0] if parents else self.history_floor
            level = base + 1 if (random.random() < self.quality and base < 6) else base
        elif op == "crossover":
            base = max(parents) if parents else self.history_floor
            level = base + 1 if (random.random() < self.quality * 0.6 and base < 6) else base
        elif op == "simplify":
            level = parents[0] if parents else self.history_floor
        else:
            level = self.history_floor
        text = _todo_html(level)
        return LLMResult(
            text=text, agent=self.name, op=op,
            tokens_in=len(prompt) // 4, tokens_out=len(text) // 4,
            latency_ms=int((time.time() - t0) * 1000),
            cost=0.0,
        )


class DummyAgent(BaseAgent):
    """段階的に進化する Todo アプリを返すダミー LLM。

    quality（0..1）が高いほど親より良いレベルを返す傾向が強い。
    op に応じて以下のように振る舞う:
        init:      L0 or L1 をランダム
        mutation:  親レベル + 1（quality の確率）/ そのまま
        crossover: max(親 A, 親 B) もしくは +1（quality の確率）
        simplify:  親レベルのまま
    """
    def __init__(self, name: str = "dummy", quality: float = 0.5):
        self.name = name
        self.quality = quality

    def _parent_levels(self, prompt: str) -> list[int]:
        return [int(m.group(1)) for m in _LEVEL_RE.finditer(prompt)]

    def call(self, prompt: str, op: str) -> LLMResult:
        t0 = time.time()
        time.sleep(0.01)
        parents = self._parent_levels(prompt)
        if op == "init":
            level = 1 if random.random() < self.quality else 0
        elif op == "mutation":
            base = parents[0] if parents else 0
            level = base + 1 if (random.random() < self.quality and base < 6) else base
        elif op == "crossover":
            base = max(parents) if parents else 0
            level = base + 1 if (random.random() < self.quality * 0.6 and base < 6) else base
        elif op == "simplify":
            level = parents[0] if parents else 0
        else:
            level = 0
        text = _todo_html(level)
        return LLMResult(
            text=text, agent=self.name, op=op,
            tokens_in=len(prompt) // 4, tokens_out=len(text) // 4,
            latency_ms=int((time.time() - t0) * 1000),
            cost=0.0,
        )


# ============================================================
# Claude エージェント（Anthropic SDK 経由）
# ============================================================
class ClaudeAgent(BaseAgent):
    """実 Claude API エージェント。

    特徴:
      - プロンプトキャッシュ（システムプロンプトを全呼び出しで共有 → 大幅コスト削減）
      - アダプティブ思考（Opus 4.7 / Opus 4.6 / Sonnet 4.6）
      - ストリーミング（max_tokens 16K で HTTP タイムアウトを回避）
      - キャッシュ込みの正確なコスト計算

    AIPL のマルチエージェント運用では claude_pool() で複数モデルを混ぜることを推奨。
    """

    # 100 万トークンあたりの料金 (USD)
    PRICING = {
        "claude-opus-4-7":   {"in": 5.00, "out": 25.00},
        "claude-opus-4-6":   {"in": 5.00, "out": 25.00},
        "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
        "claude-haiku-4-5":  {"in": 1.00, "out":  5.00},
    }

    # モデル別の既定 effort（Haiku は effort 非対応）
    DEFAULT_EFFORT = {
        "claude-opus-4-7":   "xhigh",   # 4.7 のコード/エージェント用ベスト
        "claude-opus-4-6":   "high",
        "claude-sonnet-4-6": "high",
        "claude-haiku-4-5":  None,
    }

    # モデル別の thinking サポート
    SUPPORTS_THINKING = {
        "claude-opus-4-7":   True,
        "claude-opus-4-6":   True,
        "claude-sonnet-4-6": True,
        "claude-haiku-4-5":  False,
    }

    def __init__(
        self,
        name: str,
        model: str = "claude-opus-4-7",
        max_tokens: int = 16000,
        thinking: bool | None = None,
        effort: str | None = None,
        api_key: str | None = None,
    ):
        if not _HAS_ANTHROPIC:
            raise ImportError(
                "anthropic SDK が見つかりません。`pip install anthropic` を実行してください。"
            )
        if model not in self.PRICING:
            raise ValueError(
                f"未知のモデル: {model}。サポート: {list(self.PRICING)}"
            )
        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        # None なら model 別のデフォルトを適用
        self.use_thinking = thinking if thinking is not None else self.SUPPORTS_THINKING[model]
        self.effort = effort if effort is not None else self.DEFAULT_EFFORT[model]
        self.pricing = self.PRICING[model]
        self.client = anthropic.Anthropic(api_key=api_key)

    def call(self, prompt: str, op: str) -> LLMResult:
        t0 = time.time()

        # SYSTEM_PROMPT は全呼び出しで共通なのでキャッシュ対象に
        system = [{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }]

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.use_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
            if self.effort:
                kwargs["output_config"] = {"effort": self.effort}

        try:
            # ストリーミングで HTTP タイムアウトを回避（HTML 出力は数千トークンになりがち）
            with self.client.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()
        except anthropic.APIError as e:
            # API エラーは個体の fitness=0 に倒す（GA は選択圧でこれを淘汰）
            return LLMResult(
                text=f"<!DOCTYPE html><html><body><!-- API error: {type(e).__name__} --></body></html>",
                agent=self.name, op=op,
                tokens_in=0, tokens_out=0,
                latency_ms=int((time.time() - t0) * 1000),
                cost=0.0,
            )

        # text ブロックのみ抽出（thinking ブロックは無視）
        text_parts = [b.text for b in msg.content if b.type == "text"]
        text = strip_codefence("".join(text_parts))

        # トークン使用量（キャッシュ込み）
        u = msg.usage
        in_tokens      = u.input_tokens or 0
        out_tokens     = u.output_tokens or 0
        cache_read     = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(u, "cache_creation_input_tokens", 0) or 0

        # コスト計算: cache_creation 1.25x、cache_read 0.1x、それ以外は base 価格
        in_price = self.pricing["in"]
        cost = (
            in_price * in_tokens              / 1e6 +
            in_price * cache_creation * 1.25  / 1e6 +
            in_price * cache_read     * 0.10  / 1e6 +
            self.pricing["out"] * out_tokens  / 1e6
        )

        return LLMResult(
            text=text,
            agent=self.name,
            op=op,
            # レポートには合計入力トークン（キャッシュ含む）
            tokens_in=in_tokens + cache_read + cache_creation,
            tokens_out=out_tokens,
            latency_ms=int((time.time() - t0) * 1000),
            cost=cost,
        )


# ============================================================
# OpenAI エージェント
# ============================================================
class OpenAIAgent(BaseAgent):
    """実 OpenAI API エージェント（gpt-5 系・gpt-4 系対応）。"""

    # 概算料金 (USD / 100 万トークン) — 実値は OpenAI のドキュメント参照
    PRICING = {
        "gpt-5":           {"in": 1.25, "out": 10.00},
        "gpt-5-mini":      {"in": 0.25, "out": 2.00},
        "gpt-5-nano":      {"in": 0.05, "out": 0.40},
        "gpt-5.1":         {"in": 1.25, "out": 10.00},
        "gpt-4o":          {"in": 2.50, "out": 10.00},
        "gpt-4o-mini":     {"in": 0.15, "out": 0.60},
        "gpt-4.1":         {"in": 2.00, "out": 8.00},
        "gpt-4.1-mini":    {"in": 0.40, "out": 1.60},
        "gpt-4.1-nano":    {"in": 0.10, "out": 0.40},
    }

    def __init__(self, name: str, model: str = "gpt-5-mini",
                 max_tokens: int = 16000, api_key: str | None = None):
        if not _HAS_OPENAI:
            raise ImportError("openai SDK が見つかりません。`pip install openai` を実行してください。")
        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self.pricing = self.PRICING.get(model, {"in": 1.0, "out": 5.0})
        self.client = OpenAI(api_key=api_key)

    def call(self, prompt: str, op: str) -> LLMResult:
        t0 = time.time()
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
        except openai.OpenAIError as e:
            return LLMResult(
                text=f"<!DOCTYPE html><html><body><!-- OpenAI error: {type(e).__name__} --></body></html>",
                agent=self.name, op=op, tokens_in=0, tokens_out=0,
                latency_ms=int((time.time() - t0) * 1000), cost=0.0,
            )
        text = strip_codefence(r.choices[0].message.content or "")
        u = r.usage
        in_t  = u.prompt_tokens     or 0
        out_t = u.completion_tokens or 0
        cost  = in_t * self.pricing["in"] / 1e6 + out_t * self.pricing["out"] / 1e6
        return LLMResult(
            text=text, agent=self.name, op=op,
            tokens_in=in_t, tokens_out=out_t,
            latency_ms=int((time.time() - t0) * 1000),
            cost=cost,
        )


# ============================================================
# Gemini エージェント
# ============================================================
class GeminiAgent(BaseAgent):
    """実 Google Gemini API エージェント。"""

    # 概算料金 (USD / 100 万トークン)
    PRICING = {
        "gemini-2.5-pro":         {"in": 1.25, "out": 10.00},
        "gemini-2.5-flash":       {"in": 0.30, "out": 2.50},
        "gemini-2.5-flash-lite":  {"in": 0.10, "out": 0.40},
        "gemini-3-pro-preview":   {"in": 2.00, "out": 12.00},
        "gemini-3-flash-preview": {"in": 0.40, "out": 2.50},
    }

    def __init__(self, name: str, model: str = "gemini-2.5-flash",
                 max_tokens: int = 16000, temperature: float = 0.7,
                 api_key: str | None = None):
        if not _HAS_GEMINI:
            raise ImportError("google-genai が見つかりません。`pip install google-genai` を実行してください。")
        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.pricing = self.PRICING.get(model, {"in": 1.0, "out": 5.0})
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=key)

    def call(self, prompt: str, op: str) -> LLMResult:
        t0 = time.time()
        try:
            r = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=_genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                ),
            )
        except Exception as e:
            return LLMResult(
                text=f"<!DOCTYPE html><html><body><!-- Gemini error: {type(e).__name__} --></body></html>",
                agent=self.name, op=op, tokens_in=0, tokens_out=0,
                latency_ms=int((time.time() - t0) * 1000), cost=0.0,
            )
        text = strip_codefence(r.text or "")
        um = r.usage_metadata
        in_t  = um.prompt_token_count     or 0
        out_t = um.candidates_token_count or 0
        cost  = in_t * self.pricing["in"] / 1e6 + out_t * self.pricing["out"] / 1e6
        return LLMResult(
            text=text, agent=self.name, op=op,
            tokens_in=in_t, tokens_out=out_t,
            latency_ms=int((time.time() - t0) * 1000),
            cost=cost,
        )


# ============================================================
# 他プロバイダ用の抽象テンプレート（その他拡張点）
# ============================================================
class RealAgent(BaseAgent):
    """非 Claude プロバイダを足したいとき用の抽象基底。_invoke を実装する。

    Claude を使うなら ClaudeAgent を直接使ってください。
    """
    def __init__(self, name: str, model: str, cost_per_in: float, cost_per_out: float):
        self.name = name
        self.model = model
        self.cost_per_in = cost_per_in
        self.cost_per_out = cost_per_out

    def _invoke(self, prompt: str) -> tuple[str, int, int]:
        raise NotImplementedError("非 Claude プロバイダの場合はここに SDK 呼び出しを実装")

    def call(self, prompt: str, op: str) -> LLMResult:
        t0 = time.time()
        text, t_in, t_out = self._invoke(prompt)
        text = strip_codefence(text)
        return LLMResult(
            text=text, agent=self.name, op=op,
            tokens_in=t_in, tokens_out=t_out,
            latency_ms=int((time.time() - t0) * 1000),
            cost=self.cost_per_in * t_in / 1e6 + self.cost_per_out * t_out / 1e6,
        )


# ============================================================
# エージェントプール（AIPL 多エージェント運用）
# ============================================================
class AgentPool:
    """演算子に応じてエージェントを選ぶ。AIPL の中核。"""
    def __init__(self, agents: list[BaseAgent], policy: str = "round_robin"):
        if not agents:
            raise ValueError("agents must not be empty")
        self.agents = agents
        self.policy = policy
        self._rr = 0

    def pick(self, op: str) -> BaseAgent:
        if self.policy == "round_robin":
            a = self.agents[self._rr % len(self.agents)]
            self._rr += 1
            return a
        if self.policy == "random":
            return random.choice(self.agents)
        # op_specialized: simplify は claude, mutation は gpt, crossover は gemini ... など
        if self.policy == "op_specialized":
            mapping = {
                "init": "claude", "mutation": "gpt",
                "crossover": "gemini", "simplify": "claude",
            }
            want = mapping.get(op, self.agents[0].name)
            for a in self.agents:
                if a.name == want: return a
        return self.agents[0]


def claude_pool(api_key: str | None = None, policy: str = "round_robin") -> AgentPool:
    """Claude モデル混成プール（AIPL のマルチエージェント運用の本命）。

    エージェント構成:
      - opus   : Claude Opus 4.7  — 最高知能、init/crossover で本領発揮
      - sonnet : Claude Sonnet 4.6 — バランス型、mutation の主力
      - haiku  : Claude Haiku 4.5  — 高速・低コスト、simplify など軽い操作向け

    policy="round_robin"（既定）/"random"/"op_specialized" などが利用可能。
    """
    return AgentPool([
        ClaudeAgent("opus",   model="claude-opus-4-7",   api_key=api_key),
        ClaudeAgent("sonnet", model="claude-sonnet-4-6", api_key=api_key),
        ClaudeAgent("haiku",  model="claude-haiku-4-5",  api_key=api_key),
    ], policy=policy)


def multi_vendor_pool(policy: str = "round_robin") -> AgentPool:
    """3 ベンダー混成プール (AIPL の真の ``マルチベンダーマルチエージェント'' 構成)。

    構成:
      - opus    (Claude Opus 4.7)        : 最高知能、init/crossover 向き
      - sonnet  (Claude Sonnet 4.6)      : バランス型
      - haiku   (Claude Haiku 4.5)       : 高速・低コスト
      - gpt5    (OpenAI GPT-5)           : 多様性
      - gpt5m   (OpenAI GPT-5-mini)      : 高速版
      - gemini  (Google Gemini 2.5 Pro)  : 多様性
      - flite   (Gemini 2.5 Flash-Lite)  : LOC 最少傾向 (品質モード重要)

    Quality fitness モード時、低 LOC を吐く flite が強い貢献をする。
    """
    return AgentPool([
        ClaudeAgent("opus",   model="claude-opus-4-7"),
        ClaudeAgent("sonnet", model="claude-sonnet-4-6"),
        ClaudeAgent("haiku",  model="claude-haiku-4-5"),
        OpenAIAgent("gpt5",   model="gpt-5"),
        OpenAIAgent("gpt5m",  model="gpt-5-mini"),
        GeminiAgent("gemini", model="gemini-2.5-pro"),
        GeminiAgent("flite",  model="gemini-2.5-flash-lite"),
    ], policy=policy)


def default_pool() -> AgentPool:
    """ANTHROPIC_API_KEY が設定されていれば実 Claude プール、なければダミーを返す。

    挙動を強制したい場合は claude_pool() / DummyAgent を直接呼んでください。
    """
    if _HAS_ANTHROPIC and os.environ.get("ANTHROPIC_API_KEY"):
        return claude_pool()
    return AgentPool([
        DummyAgent("claude_dummy", quality=0.7),
        DummyAgent("gpt_dummy",    quality=0.6),
        DummyAgent("gemini_dummy", quality=0.5),
    ], policy="round_robin")
