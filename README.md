# AIPL — AI Programming Loop

複数の大規模言語モデル (Claude / GPT / Gemini) をエージェントとして協調させ、進化計算 (GA) ループで問題解決を行う研究プロジェクト。

## 主な成果

- **3 仮説を統計的に検証**:
  - ① 進化速度が速い ($p=0.0016$, Cliff's $\delta = -0.81$)
  - ② 再現性が高い (SD: $0.079 \to 0.045 \to 0.000$)
  - ③ 履歴から学習する ($p=0.0054$, $\delta=-1.000$ 完全分離)
- **1 行 (2638 文字) で全 12 テストを通過する Todo アプリ**を AIPL が自動発見 (`exemplars/todo_aipl_minified_1line.aice`)
- **弱モデル単体 (Haiku) + exemplar fewshot で AIPL 混成プールと同等性能、1/3 コスト**

## クイックスタート

```bash
git clone https://github.com/yaskodama/aipl-exp.git
cd aipl-exp

# 1. Python 環境
python3 -m venv venv
venv/bin/pip install anthropic openai google-genai numpy matplotlib scipy

# 2. Playwright (テストランナー)
npm install
npx playwright install chromium

# 3. API キー設定
cp .env.example .env
# .env を編集して ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY を設定

# 4. 動作確認 (dummy LLM、API キー不要)
venv/bin/python run_pilot.py --seeds 1,2,3

# 5. 実 LLM パイロット (API キー必要)
venv/bin/python run_pilot_real.py --seeds 1,2,3 --pop 4 --gens 3
```

## ドキュメント

- **報告書 (8-11 ページ)**: `report.pdf`
- **論文ドラフト (10 ページ)**: `paper.pdf`
- **再開ガイド**: `RESUME.md`
- **厳選 .aice exemplars**: `exemplars/README.md`

## アーキテクチャ

```
┌─────────────────────────────────────────────────────┐
│  multi_vendor_pool  (Claude × 3 + GPT × 2 + Gemini × 2) │
└─────┬───────────────────────────────────────────────┘
      │ init / mutation / crossover / simplify
      ▼
┌────────────────────────────────────────────────┐
│  GA loop (ga_runner.py)                        │
│  - tournament selection + elite preservation   │
│  - .ga / .aice テキスト形式で進化過程を保存    │
└─────┬──────────────────────────────────────────┘
      │ 候補 HTML
      ▼
┌────────────────────────────────────────────────┐
│  fitness_harness.py (Playwright)               │
│  - test/quality 2 モード                       │
│  - 12 テスト × Todo / Pomodoro / 電卓          │
└────────────────────────────────────────────────┘
```

### Warm-start (履歴フィード)

```python
from warm_start import wrap_with_history, wrap_with_exemplars
from pathlib import Path

# 過去 .ga から成功変異を抽出して fewshot 注入
agent = wrap_with_history(
    ClaudeAgent("opus", model="claude-opus-4-7"),
    history_runs=[Path("runs_real/multi_s1.ga"), ...],
)

# あるいは exemplar .aice の実コードを fewshot 注入
agent = wrap_with_exemplars(
    ClaudeAgent("opus", model="claude-opus-4-7"),
    exemplar_paths=[Path("exemplars/todo_aipl_minified_1line.aice")],
)
```

## 題材 (apps/)

| Tier | 題材 | テスト数 | 状態 |
|---|---|---|---|
| T1 | 電卓 | 8 | 整備済 |
| T2 | **Todo** (主実験) | 12 | n=10 dummy + n=3 real で検証済 |
| T3 | Pomodoro | 12 | 転移実験実施中 |

## ファイル構成

```
aipl-exp/
├── apps/{calculator,todo,pomodoro}/{spec.txt,tests.spec.js}
├── exemplars/                # 厳選 .aice (1-LOC champion 等)
├── runs_real/                # 実 LLM パイロットの .ga + ゲノム
├── tools/parse_funcs.mjs     # acorn による AST 関数抽出
├── llm_agents.py             # Claude/OpenAI/Gemini Agent クラス
├── warm_start.py             # 履歴フィード
├── fitness_harness.py        # Playwright fitness 評価
├── ga_runner.py              # GA メインループ
├── gene_transition.py        # building block 分析
├── compare_runs.py           # 条件間比較 + 統計検定
├── compare_heatmaps.py       # 並列ヒートマップ
├── run_pilot*.py             # 各種パイロットランナー
├── report.tex / report.pdf   # 進捗報告書
├── paper.tex / paper.pdf     # 論文ドラフト
└── RESUME.md                 # セッション再開ガイド
```

## ライセンス

研究プロジェクト。論文公開時に明示予定。

## 参考文献

`paper.pdf` の参考文献セクションを参照。主な引用: FunSearch, Reflexion, AlphaEvolve, Eureka, AutoGen, CAMEL.
