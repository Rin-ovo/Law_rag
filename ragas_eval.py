"""
使用 RAGAS 评估 RAG 流水线。

检索阶段：context_precision, context_recall
生成阶段：faithfulness, answer_relevancy

数据格式（每行 JSON 或 HuggingFace Dataset）：
  user_input, retrieved_contexts, response, reference

API Key：请使用环境变量 OPENAI_API_KEY；下方占位符仅本地可选（勿提交真实密钥）。
可选：OPENAI_BASE_URL（兼容第三方 OpenAI 兼容网关）。

Embedding：默认使用本地 BGE-M3（与 build_faiss_index.py / FAISS 一致），路径见 --embedding-model-path 或 RAGAS_EMBEDDING_MODEL_PATH。
可选 RAGAS_EMBEDDING_BACKEND=openai 使用 OpenAI 兼容 embedding（需对应 API 与模型名）。

DeepSeek 聊天接口仅支持 n=1；RAGAS answer_relevancy 默认 strictness=3 会触发 n=3。默认将 strictness 设为 1（见 RAGAS_ANSWER_RELEVANCY_STRICTNESS）。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# ========== 可选：仅本地调试填写；公开分发请留空并使用环境变量 OPENAI_API_KEY ==========
OPENAI_API_KEY_PLACEHOLDER: str = ""
# 若使用兼容网关，可在此设置默认 base_url，或通过环境变量 OPENAI_BASE_URL
OPENAI_BASE_URL_PLACEHOLDER: str = "https://api.deepseek.com"

DEFAULT_DATA = Path(__file__).resolve().parent / "eval_data" / "ragas_test_100.jsonl"
DEFAULT_EMBEDDING_MODEL_PATH = Path(__file__).resolve().parent / "bge-m3"


def _default_embedding_backend() -> str:
    raw = os.environ.get("RAGAS_EMBEDDING_BACKEND", "local").strip().lower()
    return raw if raw in ("local", "openai") else "local"


def _answer_relevancy_strictness() -> int:
    raw = os.environ.get("RAGAS_ANSWER_RELEVANCY_STRICTNESS", "1").strip()
    try:
        n = int(raw)
    except ValueError:
        return 1
    return max(1, n)


def _apply_api_key() -> None:
    key = (OPENAI_API_KEY_PLACEHOLDER or "").strip() or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "未找到 API Key：请在环境变量 OPENAI_API_KEY 中设置，"
            "或在 ragas_eval.py 顶部填写 OPENAI_API_KEY_PLACEHOLDER。"
        )
    os.environ["OPENAI_API_KEY"] = key
    base = (OPENAI_BASE_URL_PLACEHOLDER or "").strip() or os.environ.get("OPENAI_BASE_URL")
    if base:
        os.environ["OPENAI_BASE_URL"] = base


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _validate_row(r: dict[str, Any], idx: int) -> None:
    for k in ("user_input", "retrieved_contexts", "response", "reference"):
        if k not in r:
            raise ValueError(f"第 {idx} 条缺少字段: {k}")
    ctx = r["retrieved_contexts"]
    if not isinstance(ctx, list) or not all(isinstance(x, str) for x in ctx):
        raise ValueError(f"第 {idx} 条 retrieved_contexts 必须为字符串列表")


def _wrap_langchain_embeddings(emb_lc: Any) -> Any:
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper

        return LangchainEmbeddingsWrapper(emb_lc)
    except ImportError:
        from ragas.embeddings.base import LangchainEmbeddingsWrapper

        return LangchainEmbeddingsWrapper(emb_lc)


def _build_ragas_embeddings(
    backend: str,
    embedding_model_path: Path,
) -> Any:
    backend = (backend or "local").strip().lower()
    if backend == "local":
        import torch
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings

        if not embedding_model_path.is_dir():
            raise FileNotFoundError(
                f"本地 BGE 模型目录不存在: {embedding_model_path}，"
                "请放置与 FAISS 构建时相同的 bge-m3，或通过 --embedding-model-path / RAGAS_EMBEDDING_MODEL_PATH 指定。"
            )
        device = (os.environ.get("RAGAS_EMBEDDING_DEVICE") or "").strip()
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        emb_lc = HuggingFaceBgeEmbeddings(
            model_name=str(embedding_model_path.resolve()),
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        return _wrap_langchain_embeddings(emb_lc)
    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings

        emb_model = os.environ.get("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")
        emb_lc = LCOpenAIEmbeddings(model=emb_model)
        return _wrap_langchain_embeddings(emb_lc)
    raise ValueError(f"未知 embedding 后端: {backend!r}，请使用 local 或 openai。")


def _build_ragas_llm_embeddings(
    timeout_s: int,
    max_retries: int,
    embedding_backend: str,
    embedding_model_path: Path,
) -> tuple[Any, Any]:
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    model = os.environ.get("RAGAS_LLM_MODEL", "deepseek-chat")

    llm = ChatOpenAI(
        model=model,
        temperature=0.0,
        timeout=timeout_s,
        max_retries=max_retries,
        n=1,
    )
    embeddings = _build_ragas_embeddings(embedding_backend, embedding_model_path)
    return LangchainLLMWrapper(llm), embeddings


def run_eval(
    data_path: Path,
    timeout_s: int,
    max_workers: int,
    max_retries: int,
    embedding_backend: str,
    embedding_model_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _apply_api_key()
    rows = _load_jsonl(data_path)
    for i, r in enumerate(rows):
        _validate_row(r, i)

    from datasets import Dataset

    ds = Dataset.from_list(rows)

    llm, embeddings = _build_ragas_llm_embeddings(
        timeout_s=timeout_s,
        max_retries=max_retries,
        embedding_backend=embedding_backend,
        embedding_model_path=embedding_model_path,
    )

    ar_strict = _answer_relevancy_strictness()
    try:
        from ragas import evaluate
        from ragas.metrics import (
            AnswerRelevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metrics = [
            context_precision,
            context_recall,
            faithfulness,
            AnswerRelevancy(strictness=ar_strict),
        ]
    except Exception:
        from ragas import evaluate
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        metrics = [
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy(strictness=ar_strict),
        ]

    run_config = None
    try:
        from ragas.run_config import RunConfig

        run_config = RunConfig(timeout=timeout_s, max_workers=max_workers, max_retries=max_retries)
    except Exception:
        # 兼容旧版 ragas：无 run_config 时仍可运行
        run_config = None

    eval_kwargs = dict(
        dataset=ds,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )
    if run_config is not None:
        eval_kwargs["run_config"] = run_config

    result = evaluate(**eval_kwargs)
    scores: dict[str, Any]
    if hasattr(result, "to_dict"):
        scores = result.to_dict()
    else:
        try:
            scores = dict(result)
        except Exception:
            # 兼容不同 ragas 版本的返回结构：只对数值列取均值
            df_fallback = getattr(result, "to_pandas", lambda: None)()
            if df_fallback is not None:
                import pandas as pd

                numeric_cols: list[str] = []
                for c in df_fallback.columns:
                    try:
                        as_num = pd.to_numeric(df_fallback[c], errors="coerce")
                        if as_num.notna().any():
                            numeric_cols.append(str(c))
                    except Exception:
                        continue

                if numeric_cols:
                    scores = {
                        c: float(pd.to_numeric(df_fallback[c], errors="coerce").mean())
                        for c in numeric_cols
                    }
                else:
                    scores = {"result": str(result)}
            else:
                scores = {"result": str(result)}

    stats: dict[str, Any] = {}
    df = getattr(result, "to_pandas", lambda: None)()
    if df is not None and hasattr(df, "columns") and hasattr(df, "__len__"):
        total = len(df)
        metric_stats: dict[str, dict[str, int]] = {}
        for col in df.columns:
            if col in ("user_input", "retrieved_contexts", "response", "reference"):
                continue
            try:
                valid = int(df[col].notna().sum())
                metric_stats[str(col)] = {"valid": valid, "total": total}
            except Exception:
                continue
        if metric_stats:
            stats = {"total": total, "metrics": metric_stats}
    return scores, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 评估（context_precision/recall, faithfulness, answer_relevancy）")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"JSONL 路径，默认 {DEFAULT_DATA}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("RAGAS_TIMEOUT", "120")),
        help="单次请求超时（秒），默认 120，可通过 RAGAS_TIMEOUT 覆盖",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("RAGAS_MAX_WORKERS", "4")),
        help="并发 worker 数，默认 4，可通过 RAGAS_MAX_WORKERS 覆盖",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("RAGAS_MAX_RETRIES", "2")),
        help="请求最大重试次数，默认 2，可通过 RAGAS_MAX_RETRIES 覆盖",
    )
    parser.add_argument(
        "--embedding-backend",
        type=str,
        choices=("local", "openai"),
        default=_default_embedding_backend(),
        help="embedding 来源：local=本地 BGE-M3（默认），openai=OpenAI 兼容 API（RAGAS_EMBEDDING_MODEL）",
    )
    _emb_path_env = os.environ.get("RAGAS_EMBEDDING_MODEL_PATH", "").strip()
    parser.add_argument(
        "--embedding-model-path",
        type=Path,
        default=Path(_emb_path_env) if _emb_path_env else DEFAULT_EMBEDDING_MODEL_PATH,
        help=f"本地 BGE-M3 目录（与 FAISS 一致），默认 {DEFAULT_EMBEDDING_MODEL_PATH}",
    )
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"数据文件不存在: {args.data}，可先运行 generate_ragas_testset.py")

    scores, stats = run_eval(
        data_path=args.data,
        timeout_s=args.timeout,
        max_workers=args.workers,
        max_retries=args.max_retries,
        embedding_backend=args.embedding_backend,
        embedding_model_path=args.embedding_model_path,
    )
    print("RAGAS 评估结果（均值）：")
    for k in sorted(scores.keys()):
        v = scores[k]
        print(f"  {k}: {v}")

    if stats:
        print(f"\n样本统计：total={stats['total']}")
        for col in sorted(stats["metrics"].keys()):
            item = stats["metrics"][col]
            print(f"  {col}: valid={item['valid']}/{item['total']}")


if __name__ == "__main__":
    main()
