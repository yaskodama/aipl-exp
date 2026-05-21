# セッション再開ガイド (AIPL プロジェクト)

最終コミット: `4eed0df` (Pomodoro 実験進行中)
最終更新: 2026-05-21

---

## まず最初に確認

### 1. バックグラウンド実験の進捗
```bash
cd ~/aipl-exp

# 各 Pomodoro 試行のステータス
for f in runs_real/pomo_*.ga; do
  best=$(grep -oE 'best_fitness="[0-9.]+' "$f" | tail -1)
  st=$(grep -oE 'status=("success"|"stopped")' "$f" | tail -1)
  summaries=$(grep -c "^SUMMARY " "$f")
  echo "$(basename $f): gen=$summaries $best $st"
done
```

期待される状態（実験が完走している場合）:
- `pomo_multi_s1.ga` : status=stopped or success (gen=4)
- `pomo_multi_s2.ga` : status=success (gen=4, best≥0.85)
- `pomo_multi_s3.ga` : status=stopped or success (gen=4)
- `pomo_warm_todo_s1.ga` : status=stopped or success
- `pomo_warm_todo_s2.ga` : status=stopped or success
- `pomo_warm_todo_s3.ga` : status=stopped or success

### 2. プロセスが残存しているか
```bash
ps aux | grep run_pilot_pomodoro | grep -v grep
```

---

## シナリオ別の次の手順

### シナリオ A: 全 6 試行が完走している
```bash
cd ~/aipl-exp

# 3-way 比較（pomo_multi vs pomo_warm_todo, n=3 each）
venv/bin/python compare_runs.py \
  --group pomo_multi      runs_real/pomo_multi_s[1-3].ga \
  --group pomo_warm_todo  runs_real/pomo_warm_todo_s[1-3].ga \
  --out figs/pomo_N3 \
  --ttt-threshold 0.85

# Building block heatmap
venv/bin/python compare_heatmaps.py \
  --label pomo_multi      runs_real/pomo_multi_s[1-3].ga \
  --label pomo_warm_todo  runs_real/pomo_warm_todo_s[1-3].ga \
  --out figs/pomo_N3/heatmap.png

# 結果をコミット
git add runs_real/pomo_* figs/pomo_N3/
git -c user.email="aipl@local" -c user.name="aipl-exp" \
  commit -m "Experiment A complete: Pomodoro transfer N=3 results"
git push
```

### シナリオ B: 一部の試行が失敗・ハングしている
```bash
# 残存プロセスを kill
ps aux | grep run_pilot_pomodoro | grep -v grep | awk '{print $2}' | xargs -I{} kill -9 {}

# どの試行が不完全か特定
for f in runs_real/pomo_*.ga; do
  status=$(grep -oE 'status=("success"|"stopped")' "$f" | tail -1)
  [ -z "$status" ] && echo "INCOMPLETE: $f"
done

# 不完全なファイルを退避
mv runs_real/pomo_<incomplete>.ga runs_real/pomo_<incomplete>_failed.ga
mv runs_real/pomo_<incomplete>_genomes runs_real/pomo_<incomplete>_failed_genomes

# 該当 seed のみ再実行 (fast_pool 使用)
venv/bin/python run_pilot_pomodoro.py --only <pomo_multi|pomo_warm_todo> --seeds <X>
```

### シナリオ C: バックグラウンドがまだ実行中
進捗を確認しつつ待つ。各 fast_pool 試行は約 3-5 分。

---

## プロジェクトの現状サマリ

### 検証済み仮説（dummy + real LLM 二段）

| 仮説 | dummy n=10 | real LLM n=3 |
|---|---|---|
| ① 進化速度が速い | $p=0.0016$, $\delta=-0.81$ | $\delta=-0.556$ (中-大) |
| ② 再現性が高い | SD: $0.079 \to 0.045 \to 0.000$ | SD: $0.20 \to 0.12 \to 0.11$ |
| ③ 履歴から学習する | $p=0.0054$, $\delta=-1.000$ | $\delta=-0.556$ (中-大) |

### 主要な発見

| # | 発見 | 場所 |
|---|---|---|
| 1 | **1 行 (2638 文字) で全 12 テスト合格 Todo** | `exemplars/todo_aipl_minified_1line.aice` |
| 2 | **Haiku + exemplar が multi_warm と同等性能 (1/3 コスト)** | Experiment C, runs_real/haiku_exempl_* |
| 3 | **AIPL 履歴は命名規約まで継承** (`add` 関数名の収束) | figs/real_N3/heatmap_3way.png |
| 4 | **多ベンダー多エージェント (Claude+GPT+Gemini) が真に動作** | runs_real/multi_warm_* |

### コスト履歴
- セッション累計: 約 **$7**（接続テスト + cold/multi/multi_warm + haiku_exempl + Pomodoro 部分）
- 予算 $20 のうち約 35% 使用
- 残予算: 約 **$13**

---

## 主要ファイル一覧

### コアコード
| ファイル | 役割 |
|---|---|
| `llm_agents.py` | ClaudeAgent / OpenAIAgent / GeminiAgent / multi_vendor_pool |
| `warm_start.py` | HistoryFedAgent + extract_history + exemplar_code_fewshot |
| `fitness_harness.py` | Playwright 評価（test/quality 2 モード） |
| `ga_runner.py` | GA ループ + .ga/.aice 出力 |
| `gene_transition.py` | AST building block 分析 |
| `compare_runs.py` | 条件間統計検定 |
| `compare_heatmaps.py` | 条件別ヒートマップ並列 |

### パイロットランナー
| ファイル | 用途 |
|---|---|
| `run_pilot.py` | dummy LLM パイロット (cold/warm/warmstart) |
| `run_pilot_real.py` | 実 LLM パイロット (cold/multi/multi_warm) |
| `run_pilot_compression.py` | Experiment C (Haiku+exemplar) |
| `run_pilot_pomodoro.py` | Experiment A (Pomodoro 転移) |

### ドキュメント
| ファイル | 内容 |
|---|---|
| `report.pdf` | 11 ページ進捗報告書 |
| `paper.pdf` | 10 ページ論文ドラフト + 参考文献 7 件 |
| `exemplars/README.md` | 厳選 .aice の説明 |
| `RESUME.md` | 本ファイル |

### 実験データ
| 場所 | 内容 |
|---|---|
| `runs/` (gitignore) | dummy パイロット 30 試行 |
| `runs_real/` | 実 LLM パイロット (cold/multi/multi_warm/haiku_exempl/pomo_*) |
| `exemplars/` | 厳選 .aice (1-LOC champion 含む) |
| `figs/N10/` | dummy n=10 最終分析 |
| `figs/real_N3/` | 実 LLM n=3 分析 |

---

## 残タスクと次の選択肢

### 即座に進められる
- **A 完了**: Pomodoro 転移実験 (バックグラウンド完了次第)
- **報告書/論文を A の結果で更新**: report.tex / paper.tex の §6 拡張

### 次に取り組める実験
| | 内容 | コスト | 時間 |
|---|---|---|---|
| B | 実 LLM の n=10 拡張で p<0.05 取得 | ~$9 | ~3.5h |
| D | Snake (T4) / Quiz (T5) 等の追加題材 | ~$3+ 設計時間 | ~2h+ |
| E | Threats to Validity 対策（仕様変則化）| 設計時間 | — |
| F | 1 LOC champion の品質的限界探索 | ~$2 | ~30 分 |

### 論文公開へ向けて
- arXiv 投稿 (or 学会選定)
- 著者・所属情報の確定（paper.tex の `(著者名) / (所属)` を更新）
- refs.bib の完成度確認（現在は placeholder 風の bibtex）
- リポジトリの README.md 整備（GitHub にトップレベル README が未設置）

---

## API キー (.env) について

`.env` は **`~/aipl-exp/.env`** に存在し、`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` が設定済み。GitHub には `.gitignore` で除外されているのでアップロードされていない。

新しい環境で動かす場合は `.env.example` をコピーして `.env` に API キーを書き込む。

---

## トラブルシューティング

### LLM 呼び出しがハング
- Opus 4.7 + adaptive thinking + 複雑な題材 (Pomodoro 等) で発生したケースあり
- 対処: `fast_pool` を使う (Opus 除外、thinking 無効化済み)
- 既に `run_pilot_pomodoro.py` で実装済

### Playwright テストが落ちる
- chromium バイナリが古い: `npx playwright install chromium`
- node_modules が壊れた: `rm -rf node_modules && npm install`

### Dropbox 同期衝突 (.git/)
- このプロジェクトは Dropbox 外 (`~/aipl-exp`) にあるので発生しない
- 旧 Dropbox 場所 (`/Users/kodamay/Dropbox/アプリ/site44/lecture.site44.com/logic-cir/aipl-exp/`) は読み取り専用扱い

### venv が壊れた
```bash
rm -rf venv
python3 -m venv venv
venv/bin/pip install anthropic openai google-genai numpy matplotlib scipy
```

---

## 関連リンク

- GitHub: https://github.com/yaskodama/aipl-exp
- Anthropic Console: https://console.anthropic.com/
- OpenAI Platform: https://platform.openai.com/
- Google AI Studio: https://aistudio.google.com/

EOF (RESUME.md, 約 200 行, セッション再開のための完全ガイド)
