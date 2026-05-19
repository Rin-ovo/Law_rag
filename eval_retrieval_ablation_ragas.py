#!/usr/bin/env python3
"""
检索消融 + RAGAS：对同一批 JSONL（含 user_input、reference），分别用
  - 仅稠密检索（FAISS + 重排）
  - 仅稀疏检索（BM25 + 重排）
生成 retrieved_contexts 与 response，再各跑一遍 RAGAS，输出指标便于对比。

依赖：与 fill_rag_eval_jsonl.py、ragas_eval.py 相同（本地 Qwen/BGE/FAISS、RAGAS API 等）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from fill_rag_eval_jsonl import context_string_from_ranked
from rag_qwen3_answer import answer_from_ranked, load_answer_model
from rag_search_rerank import search_with_rerank
from ragas_eval import DEFAULT_EMBEDDING_MODEL_PATH, _default_embedding_backend, run_eval

BASE_DIR = Path(__file__).resolve().parent

METHOD_META: dict[str, dict[str, str]] = {
    "dense": {
        "id": "dense",
        "label_zh": "仅稠密检索",
        "label_en": "dense_only",
        "pipeline_zh": "FAISS 语义 Top-K → 正文去重 → BGE 交叉编码器重排 → final_k",
    },
    "sparse": {
        "id": "sparse",
        "label_zh": "仅稀疏检索",
        "label_en": "sparse_only",
        "pipeline_zh": "BM25 Top-K → 正文去重 → BGE 交叉编码器重排 → final_k",
    },
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fill_rows_for_mode(
    rows: list[dict[str, Any]],
    mode: str,
    tokenizer: Any,
    model: Any,
    *,
    model_path: str,
    top_k: int,
    final_k: int,
    faiss_dir: str,
    embedding_model_path: str,
    reranker_model_path: str,
    hybrid_pool: int,
    rrf_k: int,
    start: int,
    limit: int,
) -> list[dict[str, Any]]:
    end = len(rows) if limit <= 0 else min(len(rows), start + limit)
    out: list[dict[str, Any]] = []
    if start > 0:
        out.extend(rows[:start])
    sl = rows[start:end]
    for i, r in enumerate(sl):
        r = dict(r)
        q = (r.get("user_input") or "").strip()
        if not q:
            out.append(r)
            continue
        ranked = search_with_rerank(
            q,
            top_k=top_k,
            final_k=final_k,
            faiss_dir=faiss_dir,
            embedding_model_path=embedding_model_path,
            reranker_model_path=reranker_model_path,
            retrieval_mode=mode,
            hybrid_pool=hybrid_pool,
            rrf_k=rrf_k,
        )
        r["retrieved_contexts"] = [context_string_from_ranked(x) for x in ranked]
        r["response"] = answer_from_ranked(
            q,
            ranked,
            model_path=model_path,
            tokenizer=tokenizer,
            model=model,
        )
        out.append(r)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{mode}] {start + i + 1}/{end}", flush=True)
    if end < len(rows):
        out.extend(rows[end:])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="稠密-only / 稀疏-only 检索消融，填充 JSONL 后分别跑 RAGAS",
    )
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="输入 JSONL（至少含 user_input、reference）")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=BASE_DIR / "eval_data" / "retrieval_ablation",
        help="输出目录：写入 *_filled.jsonl、*_ragas_scores.json、summary.json",
    )
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=("dense", "sparse"),
        default=("dense", "sparse"),
        help="要跑的检索模式，默认 dense sparse",
    )
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="最多处理条数，0 表示从 start 到文件末尾")
    ap.add_argument("--model-path", default=str(BASE_DIR / "Qwen3-8B"))
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--final-k", type=int, default=5)
    ap.add_argument("--faiss-dir", default=str(BASE_DIR / "faiss_index"))
    ap.add_argument("--embedding-model-path", default=str(BASE_DIR / "bge-m3"))
    ap.add_argument("--reranker-model-path", default=str(BASE_DIR / "bge-reranker-v2-m3"))
    ap.add_argument("--hybrid-pool", type=int, default=40)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--skip-ragas", action="store_true", help="只生成 filled JSONL，不调用 RAGAS API")
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("RAGAS_TIMEOUT", "120")))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("RAGAS_MAX_WORKERS", "4")))
    ap.add_argument("--max-retries", type=int, default=int(os.environ.get("RAGAS_MAX_RETRIES", "2")))
    ap.add_argument(
        "--embedding-backend",
        type=str,
        choices=("local", "openai"),
        default=_default_embedding_backend(),
    )
    _emb_env = os.environ.get("RAGAS_EMBEDDING_MODEL_PATH", "").strip()
    ap.add_argument(
        "--embedding-model-path-ragas",
        type=Path,
        default=Path(_emb_env) if _emb_env else DEFAULT_EMBEDDING_MODEL_PATH,
        help="RAGAS 评测用本地 BGE 路径（与 FAISS 一致即可）",
    )
    args = ap.parse_args()

    if not args.inp.is_file():
        print(f"输入文件不存在: {args.inp}", file=sys.stderr)
        sys.exit(1)

    rows = _load_jsonl(args.inp)
    if not rows:
        print("JSONL 为空", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r} | "
        f"torch.cuda.is_available()={torch.cuda.is_available()}",
        flush=True,
    )
    print(f"加载 Qwen（一次）: {args.model_path}", flush=True)
    tokenizer, model = load_answer_model(args.model_path, use_fp16=True)

    summary: dict[str, Any] = {"input": str(args.inp), "modes": {}}

    for mode in args.modes:
        print(f"\n=== 检索模式: {mode}（仅 FAISS 或仅 BM25，均经同一重排器）===", flush=True)
        filled = fill_rows_for_mode(
            rows,
            mode,
            tokenizer,
            model,
            model_path=args.model_path,
            top_k=args.top_k,
            final_k=args.final_k,
            faiss_dir=args.faiss_dir,
            embedding_model_path=args.embedding_model_path,
            reranker_model_path=args.reranker_model_path,
            hybrid_pool=args.hybrid_pool,
            rrf_k=args.rrf_k,
            start=args.start,
            limit=args.limit,
        )
        filled_path = args.out_dir / f"ablation_{mode}_filled.jsonl"
        _write_jsonl(filled_path, filled)
        print(f"已写入: {filled_path}", flush=True)

        mode_result: dict[str, Any] = {"filled_jsonl": str(filled_path.resolve())}

        if not args.skip_ragas:
            print(f"运行 RAGAS: {filled_path}", flush=True)
            scores, stats = run_eval(
                data_path=filled_path,
                timeout_s=args.timeout,
                max_workers=args.workers,
                max_retries=args.max_retries,
                embedding_backend=args.embedding_backend,
                embedding_model_path=args.embedding_model_path_ragas,
            )
            scores_path = args.out_dir / f"ablation_{mode}_ragas_scores.json"
            with open(scores_path, "w", encoding="utf-8") as f:
                json.dump({"scores": scores, "stats": stats}, f, ensure_ascii=False, indent=2)
            print("RAGAS 均值:", flush=True)
            for k in sorted(scores.keys()):
                print(f"  {k}: {scores[k]}", flush=True)
            print(f"指标已写入: {scores_path}", flush=True)
            mode_result["ragas_scores"] = scores
            mode_result["ragas_stats"] = stats
            mode_result["scores_json"] = str(scores_path.resolve())
        else:
            mode_result["ragas_skipped"] = True

        summary["modes"][mode] = mode_result

    summary_path = args.out_dir / "retrieval_ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
