# AIPL Exemplars — 進化の成果物として保存された .aice 実装

このディレクトリには、AIPL パイロットで実際に発見・進化された優秀な実装、および新題材用の参照実装を厳選して収めています。各 `.aice` ファイルは単一 HTML / JS / CSS で、Playwright テストでそれぞれの仕様を満たすことを再確認済みです。

## ファイル一覧（実証ベンチマーク）

| ファイル | 題材 | LOC | 文字数 | fitness (quality) | テスト | 由来 |
|---|---|---|---|---|---|---|
| `todo_aipl_minified_1line.aice` | Todo | **1** | 2,588 | **0.9986** | 12/12 | multi_warm s3 g3i1, gpt5_warm simplify (champion) |
| `todo_aipl_compact_18lines.aice` | Todo | 18 | 4,908 | 0.9748 | 12/12 | multi_warm s2 g3i1, gpt5_warm mutation |
| `todo_aipl_balanced_37lines.aice` | Todo | 38 | 5,537 | 0.9468 | 12/12 | multi_nohist s2 g1i1, gpt5m simplify |
| `todo_haiku_baseline.aice` | Todo | 55 | 6,091 | 0.9230 | 12/12 | cold s3 g3i1 (Haiku 単体ベースライン) |
| `pomodoro_reference.aice` | Pomodoro | 117 | 4,144 | 0.8362 | 12/12 | 手書き参照実装（T3 転移実験の出発点） |

## 用途

### 1. Warm-start 用の history seed
`warm_start.py::wrap_with_history()` の引数として、これらの実装が含まれる過去試行 `.ga` を渡すと、新たな LLM 呼び出しのプロンプトに少数例として注入される。

### 2. ベンチマーク基準値
新しいエージェント構成や fitness 関数を試すとき、これらの数値（特に Todo の 1 行版 fit=0.9986）が **AIPL が達成可能な性能の上限**として参照できる。

### 3. 教材 / 論文の補足資料
``LLM 出力 277-415 LOC → AIPL 後 1 LOC''という極端な最適化事例を、生のコードで示せる。

### 4. 転移実験の起点
`pomodoro_reference.aice` は Todo で得た知見を Pomodoro に転移する実験の起点。手書きで仕様を満たす最小限の参照実装。

## 各ファイルの特徴

### `todo_aipl_minified_1line.aice` — チャンピオン
2,638 文字すべてを 1 行に圧縮。改行・空白を一切含まない高密度実装。CSS は最小限の `text-decoration: line-through` のみ、JS は IIFE で `(()=>{const k='todos.v1',fk='todos.filter',$=s=>document.querySelector(s),...})()` の極短コーディングスタイル。全 12 機能（CRUD・完了トグル・編集・3 種フィルタ・localStorage・clear-completed）を実装。

### `todo_aipl_compact_18lines.aice` — 実用的圧縮版
1 行版は可読性ゼロだが、こちらは 18 行で各機能ブロックが論理的に分離されている。コードを理解しつつ短さも欲しい場合の選択肢。

### `todo_aipl_balanced_37lines.aice` — バランス版
LLM 単体出力（277-415 LOC）から約 1/10 に圧縮しつつ、宣言と関数を読みやすい形で配置。プロダクションコードに最も近い書き味。

### `todo_haiku_baseline.aice` — Haiku ベースライン
Single LLM (cold 条件) で到達した最良結果。AIPL なしでも 55 LOC で 12/12 通過するという好結果だが、AIPL multi_warm はさらに 55 倍の圧縮を達成。

### `pomodoro_reference.aice` — Pomodoro 参照実装
T3 (Pomodoro タイマー) の手書き参照。setInterval + 仮想時計連携を考慮した実装で、12/12 テストを通過。今後 Pomodoro 題材で AIPL を回す際の "正解の一つ" として参照可能。

## 再評価方法

```bash
cd ~/aipl-exp
for f in exemplars/*.aice; do
  app=$(echo $f | grep -q pomodoro && echo pomodoro || echo todo)
  venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from fitness_harness import evaluate
print('$f:', evaluate(open('$f').read(), app='$app', fitness_mode='quality'))
"
done
```

## 関連ドキュメント
- 抽出元: `runs_real/`
- 統計分析: `figs/real_N3/`
- 詳細: `paper.pdf` §6 (実 LLM 検証) / `report.pdf` §9
