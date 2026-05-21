/**
 * tools/parse_funcs.mjs — acorn による関数抽出（AST ベース）
 *
 * 使い方:
 *   echo "<html>...</html>" | node tools/parse_funcs.mjs
 *   または:
 *   node tools/parse_funcs.mjs --file path/to/genome.aice
 *
 * 出力 (stdout, JSON):
 *   {
 *     "functions": [
 *       {"name": "addTodo", "body_normalized": "...", "type": "FunctionDeclaration"},
 *       ...
 *     ],
 *     "errors": [...]
 *   }
 *
 * regex 版 (gene_transition.py 内) より以下に強い:
 *   - メソッド定義 (class M { add() {} })
 *   - オブジェクトメソッド ({ add() {} } 短縮形)
 *   - ネストした関数のスコープ識別
 *   - 文字列/コメント内の偽 hit を排除
 */

import * as acorn from "acorn";
import { readFileSync } from "fs";

const args = process.argv.slice(2);
let html = "";
const fileIdx = args.indexOf("--file");
if (fileIdx >= 0 && args[fileIdx + 1]) {
  html = readFileSync(args[fileIdx + 1], "utf-8");
} else {
  html = readFileSync(0, "utf-8");
}

function extractScripts(s) {
  const out = [];
  const re = /<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi;
  let m;
  while ((m = re.exec(s)) !== null) out.push(m[1]);
  return out;
}

function normalize(s) {
  return s.replace(/\s+/g, " ").trim();
}

function walkAndCollect(node, source, results, parentName) {
  if (!node || typeof node !== "object") return;

  const tryRecord = (name, bodyNode, type) => {
    if (!bodyNode || bodyNode.start == null || bodyNode.end == null) return;
    const body = source.slice(bodyNode.start, bodyNode.end);
    results.push({ name, body_normalized: normalize(body), type });
  };

  switch (node.type) {
    case "FunctionDeclaration":
      if (node.id && node.id.name)
        tryRecord(node.id.name, node.body, "FunctionDeclaration");
      break;
    case "VariableDeclarator":
      if (node.id && node.id.name && node.init) {
        const init = node.init;
        if (init.type === "FunctionExpression" || init.type === "ArrowFunctionExpression") {
          tryRecord(node.id.name, init.body, init.type);
        }
      }
      break;
    case "MethodDefinition":
      if (node.key && node.key.name)
        tryRecord(node.key.name, node.value && node.value.body, "MethodDefinition");
      break;
    case "Property":
      // { addTodo() { ... } } 短縮 / { addTodo: function() {} }
      if (node.key && node.key.name && node.value &&
          (node.value.type === "FunctionExpression" || node.value.type === "ArrowFunctionExpression")) {
        tryRecord(node.key.name, node.value.body, node.value.type);
      }
      break;
    case "AssignmentExpression":
      // foo.bar = function() {} / foo.bar = () => {}
      if (node.left && node.left.type === "MemberExpression" &&
          node.left.property && node.left.property.name && node.right &&
          (node.right.type === "FunctionExpression" || node.right.type === "ArrowFunctionExpression")) {
        tryRecord(node.left.property.name, node.right.body, node.right.type);
      }
      break;
  }

  for (const key in node) {
    if (key === "loc" || key === "range" || key === "start" || key === "end" || key === "type") continue;
    const v = node[key];
    if (Array.isArray(v)) {
      for (const c of v) walkAndCollect(c, source, results, parentName);
    } else if (v && typeof v === "object") {
      walkAndCollect(v, source, results, parentName);
    }
  }
}

const results = [];
const errors = [];

for (const script of extractScripts(html)) {
  try {
    const ast = acorn.parse(script, {
      ecmaVersion: "latest", sourceType: "script", allowReturnOutsideFunction: true,
    });
    walkAndCollect(ast, script, results, null);
  } catch (err) {
    // モジュール形式で再試行
    try {
      const ast = acorn.parse(script, {
        ecmaVersion: "latest", sourceType: "module", allowReturnOutsideFunction: true,
      });
      walkAndCollect(ast, script, results, null);
    } catch (err2) {
      errors.push({ message: err2.message, pos: err2.pos });
    }
  }
}

// data-testid も併せて出す（regex で十分だが情報をまとめる）
const testids = [];
{
  const re = /data-testid\s*=\s*["']([A-Za-z_][\w-]*)["']/g;
  let m;
  while ((m = re.exec(html)) !== null) testids.push(m[1]);
}

process.stdout.write(JSON.stringify({
  functions: results,
  testids: Array.from(new Set(testids)),
  errors,
}));
