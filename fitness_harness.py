"""
fitness ハーネス: 候補 HTML → Playwright テスト実行 → fitness を返す

使い方:
    from fitness_harness import evaluate
    r = evaluate(candidate_html_str, app="todo")
    print(r["fitness"], r["passed"], "/", r["total"])

要件:
    cd aipl-exp/
    npm install
    npx playwright install chromium
"""

from __future__ import annotations
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).parent
LOC_THRESHOLD = 500   # この行数で LOC ペナルティが 0 になる

APP_CONFIGS = {
    "calculator": {
        "test_file": "apps/calculator/tests.spec.js",
        "total_tests": 8,
    },
    "todo": {
        "test_file": "apps/todo/tests.spec.js",
        "total_tests": 12,
    },
    "pomodoro": {
        "test_file": "apps/pomodoro/tests.spec.js",
        "total_tests": 12,
    },
}


@dataclass
class FitnessResult:
    fitness: float
    test_pass_rate: float
    passed: int
    total: int
    loc: int
    loc_score: float
    failed_tests: list[str]
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_playwright_json(stdout: str) -> dict | None:
    """Playwright の --reporter=json は最後に JSON を吐く。前後にノイズがあっても拾う。"""
    m = re.search(r"\{[\s\S]*\}\s*$", stdout)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _collect_specs(node, out: list):
    """Playwright JSON のネストした suites をフラットにする"""
    if isinstance(node, dict):
        if "specs" in node:
            out.extend(node["specs"])
        for s in node.get("suites", []):
            _collect_specs(s, out)


def evaluate(candidate_html: str, app: str = "todo", timeout: int = 120,
             fitness_mode: str = "test") -> FitnessResult:
    """候補 HTML を Playwright で評価して fitness を返す。

    fitness_mode:
      - "test"    (既定): fitness = 0.85*pass_rate + 0.15*loc_score (テスト合格重視)
      - "quality"      : fitness = pass_rate * (0.3 + 0.7*loc_score) (LOC 簡潔さを強く重視)
        テストが落ちると quality fitness は急減 (multiplicative gating)。
    """
    if app not in APP_CONFIGS:
        raise ValueError(f"unknown app: {app}")
    cfg = APP_CONFIGS[app]
    total_expected = cfg["total_tests"]

    with tempfile.NamedTemporaryFile(
        suffix=".html", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(candidate_html)
        candidate_path = f.name

    try:
        env = {**os.environ, "CANDIDATE_HTML": candidate_path}
        proc = subprocess.run(
            ["npx", "playwright", "test", cfg["test_file"], "--reporter=json"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout,
        )
        report = _extract_playwright_json(proc.stdout)
        if not report:
            return FitnessResult(
                fitness=0.0, test_pass_rate=0.0, passed=0, total=total_expected,
                loc=candidate_html.count("\n") + 1, loc_score=0.0,
                failed_tests=[], error=f"could not parse JSON. stderr={proc.stderr[:300]}",
            )

        specs: list = []
        _collect_specs(report, specs)
        passed = sum(1 for s in specs if s.get("ok"))
        total = len(specs) or total_expected
        failed = [s.get("title", "?") for s in specs if not s.get("ok")]

        pass_rate = passed / total if total else 0.0
        loc = candidate_html.count("\n") + 1
        loc_score = max(0.0, 1 - loc / LOC_THRESHOLD)
        if fitness_mode == "quality":
            fitness = pass_rate * (0.3 + 0.7 * loc_score)
        else:
            fitness = 0.85 * pass_rate + 0.15 * loc_score

        return FitnessResult(
            fitness=round(fitness, 4),
            test_pass_rate=round(pass_rate, 4),
            passed=passed, total=total,
            loc=loc, loc_score=round(loc_score, 4),
            failed_tests=failed,
        )
    except subprocess.TimeoutExpired:
        return FitnessResult(
            fitness=0.0, test_pass_rate=0.0, passed=0, total=total_expected,
            loc=candidate_html.count("\n") + 1, loc_score=0.0,
            failed_tests=[], error="timeout",
        )
    finally:
        try:
            os.unlink(candidate_path)
        except OSError:
            pass


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("html_file")
    p.add_argument("--app", default="todo", choices=list(APP_CONFIGS))
    a = p.parse_args()
    text = Path(a.html_file).read_text(encoding="utf-8")
    r = evaluate(text, app=a.app)
    print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
