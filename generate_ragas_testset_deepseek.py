#!/usr/bin/env python3
"""
使用 DeepSeek（OpenAI 兼容接口）按四类提示词从本库 markdown 法条中采样上下文，
生成 RAGAS 评测用 JSONL：每行含 user_input、reference、retrieved_contexts=[]、response=""。

API Key：请使用环境变量 OPENAI_API_KEY；下方占位符仅本地调试可选填写（勿提交真实密钥到公开仓库）。

默认总量 400 条，分布：事实 150（50 次×3）、解释 100（50×2）、案例 100、比较 50。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

# ========== 可选：仅本地调试填写；公开分发请留空并使用环境变量 OPENAI_API_KEY ==========
OPENAI_API_KEY_PLACEHOLDER: str = ""
OPENAI_BASE_URL_PLACEHOLDER: str = "https://api.deepseek.com"

H3_LINE = re.compile(r"^### \[([^\]]+)\]\s*(.+?)\s*$")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MD_DIR = BASE_DIR / "markdown"
DEFAULT_OUT = BASE_DIR / "eval_data" / "ragas_deepseek_400.jsonl"


def _parse_parent_lines(md_text: str) -> list[tuple[str, str]]:
    """返回 (law_key, 整行含 ###)。"""
    out: list[tuple[str, str]] = []
    for line in md_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = H3_LINE.match(line)
        if m:
            out.append((m.group(1).strip(), line))
    return out


def load_corpus_parents(md_dir: Path, min_body_len: int = 25) -> list[tuple[str, str]]:
    """收集所有合格 parent 行（正文片段不宜过短，减少只有章名的行）。"""
    rows: list[tuple[str, str]] = []
    for p in sorted(md_dir.glob("*.md")):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        for law_key, parent_line in _parse_parent_lines(text):
            m = H3_LINE.match(parent_line)
            body = (m.group(2) if m else "").strip()
            if len(body) < min_body_len:
                continue
            rows.append((law_key, parent_line))
    return rows


def _deepseek_client():
    from openai import OpenAI

    api_key = (OPENAI_API_KEY_PLACEHOLDER or "").strip() or (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "未找到 API Key：请设置环境变量 OPENAI_API_KEY，或在 generate_ragas_testset_deepseek.py 顶部填写 OPENAI_API_KEY_PLACEHOLDER（仅本地）。"
        )
    base_url = (
        (OPENAI_BASE_URL_PLACEHOLDER or "").strip()
        or (os.environ.get("OPENAI_BASE_URL") or "").strip()
        or "https://api.deepseek.com"
    ).rstrip("/")
    return OpenAI(api_key=api_key, base_url=base_url)


def _chat_json_array(client: Any, model: str, system: str, user: str, max_retries: int = 5) -> list[dict[str, Any]]:
    last_err: str | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.4,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = _extract_json_array(raw)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
            last_err = "模型返回非 JSON 数组"
        except Exception as e:
            last_err = str(e)
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"DeepSeek 调用失败（重试 {max_retries} 次）: {last_err}")


def _extract_json_array(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        return data
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        return json.loads(m.group(0))
    raise ValueError("无法解析 JSON 数组")


def _normalize_qa_item(obj: dict[str, Any]) -> tuple[str, str] | None:
    q = (obj.get("user_input") or obj.get("question") or "").strip()
    ref = (obj.get("reference") or obj.get("参考") or obj.get("ground_truth") or "").strip()
    if not q or not ref:
        return None
    return q, ref


PROMPT_FACT = """任务：基于以下法律文本，生成恰好 {n} 个事实型问答对。要求：
1. 问题必须有明确答案，答案须直接来源于给定文本（可摘抄或轻微改写，不得编造文本中不存在的数字、主体或期限）。
2. 侧重：期限、主体资格、数额、程序节点、定义用语等可核验信息。
3. 仅输出一个 JSON 数组，不要 Markdown、不要解释。数组中每个元素为对象，且仅含两个键："user_input" 与 "reference"（字符串）。

法律文本：
{law_text}
"""

PROMPT_EXPLAIN = """任务：基于以下法律文本，生成恰好 {n} 个解释型问题。要求：
1. 问题应使用「为什么」「如何理解」「立法目的」「适用范围」等咨询式表述。
2. 答案需概括、阐释条文中的法律概念或制度，不是简单照抄一句；仍须以该文本为依据，不得引入文本外具体事实。
3. 仅输出 JSON 数组，元素为 {{"user_input": "...", "reference": "..."}}。

法律文本：
{law_text}
"""

PROMPT_CASE = """任务：将以下法律条文改写为恰好 1 个具体短案例咨询。要求：
1. 虚构简单人物与情节，与条文制度相关；提问应涉及是否违法、责任归属或如何维权等。
2. reference 须结合案情给出结论与理由，并引用给定条文中的规则（可摘抄关键句）；不得引用未提供的法条编号以外的外部法条。
3. 仅输出 JSON 数组，长度为 1，元素为 {{"user_input": "...", "reference": "..."}}。

法律文本：
{law_text}
"""

PROMPT_COMPARE = """任务：结合以下两段法律文本（文本A、文本B），生成恰好 1 个比较或冲突分析型问题。要求：
1. 问题须同时涉及两段文本中的制度或主体，要求回答时综合两段材料（可在问题里明确「结合两段材料」）。
2. reference 中分述两段条文要点并给出对比结论；不得编造两段文本中不存在的规定。
3. 仅输出 JSON 数组，长度为 1，元素为 {{"user_input": "...", "reference": "..."}}。

文本A：
{law_a}

文本B：
{law_b}
"""

SYSTEM = (
    "你是中文法律评测数据编写助手。你必须只输出合法 UTF-8 JSON（一个数组），"
    "键名严格使用 user_input 与 reference。不要输出任何其它文字。"
)


def _pick_distinct_parents(pool: list[tuple[str, str]], rng: random.Random, k: int) -> list[tuple[str, str]]:
    if len(pool) < k:
        raise RuntimeError(f"语料 parent 行不足：需要至少 {k} 条，实际 {len(pool)}")
    return rng.sample(pool, k)


def generate_dataset(
    *,
    md_dir: Path,
    out_path: Path,
    rng: random.Random,
    model: str,
    n_fact_calls: int,
    n_explain_calls: int,
    n_case_calls: int,
    n_compare_calls: int,
    resume: bool,
) -> None:
    pool = load_corpus_parents(md_dir)
    if len(pool) < 10:
        raise RuntimeError(f"{md_dir} 下可用法条过少，请检查 markdown 语料。")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_q: set[str] = set()
    if resume and out_path.is_file():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    seen_q.add((row.get("user_input") or "").strip())
                except json.JSONDecodeError:
                    continue

    client = _deepseek_client()
    file_mode = "a" if (resume and out_path.exists()) else "w"

    def write_rows(fh: Any, items: list[tuple[str, str]]) -> int:
        n = 0
        for q, ref in items:
            if q in seen_q:
                continue
            seen_q.add(q)
            rec = {
                "user_input": q,
                "reference": ref,
                "retrieved_contexts": [],
                "response": "",
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
        return n

    total_written = 0
    with open(out_path, file_mode, encoding="utf-8") as out_f:
        # 1) 事实型：每次 3 条
        for i in range(n_fact_calls):
            (_, parent) = pool[rng.randrange(len(pool))]
            user = PROMPT_FACT.format(n=3, law_text=parent)
            arr = _chat_json_array(client, model, SYSTEM, user)
            good: list[tuple[str, str]] = []
            for obj in arr:
                if not isinstance(obj, dict):
                    continue
                pair = _normalize_qa_item(obj)
                if pair:
                    good.append(pair)
            good = good[:3]
            if len(good) < 3:
                print(f"[warn] 事实型 第{i+1}次 仅得到 {len(good)} 条有效项，仍写入", file=sys.stderr)
            n = write_rows(out_f, good)
            total_written += n
            print(f"fact batch {i+1}/{n_fact_calls} -> +{n} (累计约 {total_written})")

        # 2) 解释型：每次 2 条
        for i in range(n_explain_calls):
            (_, parent) = pool[rng.randrange(len(pool))]
            user = PROMPT_EXPLAIN.format(n=2, law_text=parent)
            arr = _chat_json_array(client, model, SYSTEM, user)
            good = []
            for obj in arr:
                if isinstance(obj, dict):
                    pair = _normalize_qa_item(obj)
                    if pair:
                        good.append(pair)
            good = good[:2]
            if len(good) < 2:
                print(f"[warn] 解释型 第{i+1}次 仅得到 {len(good)} 条", file=sys.stderr)
            n = write_rows(out_f, good)
            total_written += n
            print(f"explain batch {i+1}/{n_explain_calls} -> +{n}")

        # 3) 案例型：每次 1 条
        for i in range(n_case_calls):
            (_, parent) = pool[rng.randrange(len(pool))]
            user = PROMPT_CASE.format(law_text=parent)
            arr = _chat_json_array(client, model, SYSTEM, user)
            good = []
            for obj in arr:
                if isinstance(obj, dict):
                    pair = _normalize_qa_item(obj)
                    if pair:
                        good.append(pair)
            good = good[:1]
            if not good:
                print(f"[warn] 案例型 第{i+1}次 无有效项，跳过", file=sys.stderr)
                continue
            n = write_rows(out_f, good)
            total_written += n
            print(f"case batch {i+1}/{n_case_calls} -> +{n}")

        # 4) 比较型：两段文本
        for i in range(n_compare_calls):
            a, b = _pick_distinct_parents(pool, rng, 2)
            if a[0] == b[0]:
                continue
            user = PROMPT_COMPARE.format(law_a=a[1], law_b=b[1])
            arr = _chat_json_array(client, model, SYSTEM, user)
            good = []
            for obj in arr:
                if isinstance(obj, dict):
                    pair = _normalize_qa_item(obj)
                    if pair:
                        good.append(pair)
            good = good[:1]
            if not good:
                print(f"[warn] 比较型 第{i+1}次 无有效项，跳过", file=sys.stderr)
                continue
            n = write_rows(out_f, good)
            total_written += n
            print(f"compare batch {i+1}/{n_compare_calls} -> +{n}")

    print(f"完成。新写入约 {total_written} 条到 {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek 生成 RAGAS 格式测试集 JSONL")
    ap.add_argument("--md-dir", type=Path, default=DEFAULT_MD_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-fact-calls", type=int, default=50, help="每次生成 3 条，默认 50 次 -> 150 条")
    ap.add_argument("--n-explain-calls", type=int, default=50, help="每次 2 条 -> 100 条")
    ap.add_argument("--n-case-calls", type=int, default=100)
    ap.add_argument("--n-compare-calls", type=int, default=50)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="以追加方式打开输出文件，并跳过已存在的 user_input；仍会重新请求全部批次 API，适合断网后手工删去坏行再补跑。",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    generate_dataset(
        md_dir=args.md_dir,
        out_path=args.out,
        rng=rng,
        model=args.model,
        n_fact_calls=args.n_fact_calls,
        n_explain_calls=args.n_explain_calls,
        n_case_calls=args.n_case_calls,
        n_compare_calls=args.n_compare_calls,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
