#!/usr/bin/env python3
"""
读取 RAGAS 格式 JSONL（含 user_input、reference；retrieved_contexts/response 可为空），
对每条调用本仓库检索 + Qwen 生成，写入 retrieved_contexts 与 response。

retrieved_contexts 的字符串格式与 eval_data/10.jsonl 一致：`### [law_key] 条文标题/首行摘要`
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# 在导入 rag 模块之前设置，避免其 setdefault 选错卡；与两文件内默认一致为 GPU 0。
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from rag_qwen3_answer import answer_from_ranked, load_answer_model
from rag_search_rerank import search_with_rerank

H3_LINE = re.compile(r"^### \[([^\]]+)\]\s*(.*)$")

BASE_DIR = Path(__file__).resolve().parent


def context_string_from_ranked(r: dict[str, Any]) -> str:
    meta = r.get("metadata") or {}
    law_key = (meta.get("law_key") or "").strip()
    body = (meta.get("Law_Article") or meta.get("Law_Section") or "").strip()
    parent = (meta.get("parent_content") or "").strip()
    if not law_key and parent:
        first = parent.split("\n", 1)[0].strip()
        m = H3_LINE.match(first)
        if m:
            law_key = m.group(1).strip()
            body = body or m.group(2).strip()
    if law_key:
        if body:
            return f"### [{law_key}] {body}"
        return f"### [{law_key}]"
    return (r.get("page_content") or "").strip()[:2000]


def main() -> None:
    ap = argparse.ArgumentParser(description="用本地 RAG 填充 JSONL 的 retrieved_contexts 与 response")
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-path", default=str(BASE_DIR / "Qwen3-8B"))
    ap.add_argument("--start", type=int, default=0, help="从第几条开始（0-based）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理条数，0 表示全部")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--final-k", type=int, default=5)
    ap.add_argument("--faiss-dir", default=str(BASE_DIR / "faiss_index"))
    ap.add_argument("--embedding-model-path", default=str(BASE_DIR / "bge-m3"))
    ap.add_argument("--reranker-model-path", default=str(BASE_DIR / "bge-reranker-v2-m3"))
    ap.add_argument(
        "--no-hybrid",
        action="store_true",
        help="关闭 BM25+RRF，仅用语义检索（与旧行为一致）",
    )
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF 融合常数 k，默认 60")
    ap.add_argument("--hybrid-pool", type=int, default=40, help="混合检索合并后送入重排的最大候选数")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    end = len(rows) if args.limit <= 0 else min(len(rows), args.start + args.limit)
    sl = rows[args.start : end]
    if not sl:
        print("没有可处理的行", file=sys.stderr)
        sys.exit(1)

    print(f"加载问答模型（仅一次）: {args.model_path}")
    print(
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r} | "
        f"torch.cuda.is_available()={torch.cuda.is_available()} | "
        f"device_count={torch.cuda.device_count()}",
        flush=True,
    )
    if torch.cuda.is_available():
        print(f"当前 GPU: {torch.cuda.get_device_name(0)}", flush=True)
    tokenizer, model = load_answer_model(args.model_path, use_fp16=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out_f:
        # 先写出未处理的前缀（若有 --start>0，从原文件复制）
        if args.start > 0:
            for r in rows[: args.start]:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for i, r in enumerate(sl):
            q = (r.get("user_input") or "").strip()
            if not q:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue
            ranked = search_with_rerank(
                q,
                top_k=args.top_k,
                final_k=args.final_k,
                faiss_dir=args.faiss_dir,
                embedding_model_path=args.embedding_model_path,
                reranker_model_path=args.reranker_model_path,
                use_hybrid=not args.no_hybrid,
                rrf_k=args.rrf_k,
                hybrid_pool=args.hybrid_pool,
            )
            r["retrieved_contexts"] = [context_string_from_ranked(x) for x in ranked]
            r["response"] = answer_from_ranked(
                q,
                ranked,
                model_path=args.model_path,
                tokenizer=tokenizer,
                model=model,
            )
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()
            if (i + 1) % 10 == 0 or i == 0:
                print(f"进度 {args.start + i + 1}/{len(rows)}", flush=True)
        # 若只处理片段，尾部未重跑的样本从原 rows 写回
        if end < len(rows):
            for r in rows[end:]:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
