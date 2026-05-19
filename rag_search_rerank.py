import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS


# 默认物理 GPU 0；单卡机误用 "1" 会导致无可见 GPU。多卡请 export CUDA_VISIBLE_DEVICES=… 再启动。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# 同一进程内复用 FAISS 与重排器，避免每条 query 重复 load 模型（fill JSONL 时否则会加载数百次）
_FAISS_BUNDLE_CACHE: dict[tuple[str, str, str], tuple[Any, Any]] = {}
_RERANKER_CACHE: dict[tuple[str, bool], Any] = {}
# BM25 与 FAISS 共用同一批 Document 顺序；首次构建需遍历全库，略慢一次
_BM25_BUNDLE_CACHE: dict[tuple[str, str, str], tuple[Any, list[Any]]] = {}


def _embedding_device() -> str:
    """
    嵌入模型所在设备。可通过环境变量覆盖，避免与同进程中的大模型争显存：
      RAG_EMBEDDING_DEVICE=cpu   # 推荐：Qwen 已占满 GPU 时检索用 CPU
      RAG_EMBEDDING_DEVICE=cuda  # 仅检索、不加载大模型时可用
    未设置时：有 CUDA 则用 cuda，否则 cpu。
    """
    raw = (os.environ.get("RAG_EMBEDDING_DEVICE") or "").strip().lower()
    if raw == "cpu":
        return "cpu"
    if raw == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_embeddings(model_path: str):
    device_str = _embedding_device()
    return HuggingFaceBgeEmbeddings(
        model_name=model_path,
        model_kwargs={"device": device_str},
        encode_kwargs={"normalize_embeddings": True},
    )


def _normalize_passage_for_dedup(text: str) -> str:
    """用于去重：去首尾空白并压缩连续空白，避免仅换行不同的重复。"""
    s = (text or "").strip()
    return re.sub(r"\s+", " ", s)


def dedupe_candidates_by_content(candidates: List[object]) -> List[object]:
    """
    检索阶段：按规范化正文去重，保留先出现的（向量相似度顺序），减少重排无效计算。
    """
    seen: set[str] = set()
    out: List[object] = []
    for d in candidates:
        raw = getattr(d, "page_content", "") or ""
        key = _normalize_passage_for_dedup(raw)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def dedupe_ranked_top_k(ranked: List[Dict[str, Any]], final_k: int) -> List[Dict[str, Any]]:
    """
    重排后：按分数从高到低遍历，取前 final_k 条「正文不重复」的结果（同文保留分更高的一条）。
    """
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for r in ranked:
        key = _normalize_passage_for_dedup((r.get("page_content") or ""))
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= final_k:
            break
    return out


def _maybe_move_faiss_index_to_gpu(db: FAISS, device: int = 0) -> None:
    """
    将 LangChain FAISS wrapper 内部的 index 从 CPU 迁移到 GPU。
    如果 faiss-gpu 不可用/无 GPU/迁移失败，会自动回退到 CPU。
    """
    try:
        import faiss  # type: ignore
    except Exception:
        return

    try:
        if getattr(faiss, "get_num_gpus", None) is None or faiss.get_num_gpus() <= 0:
            return
        if not torch.cuda.is_available():
            return
    except Exception:
        return

    try:
        # 标准做法：StandardGpuResources + index_cpu_to_gpu
        res = faiss.StandardGpuResources()
        db.index = faiss.index_cpu_to_gpu(res, device, db.index)
    except Exception as e:
        # 不要让 GPU 迁移失败阻断主流程
        print(f"[warn] move faiss index to gpu failed, fallback to cpu: {e}")


def _get_faiss_bundle(faiss_dir: str, embedding_model_path: str) -> tuple[Any, Any]:
    """返回 (embeddings, db)，按目录 + 嵌入路径 + 设备缓存。"""
    key = (
        os.path.abspath(faiss_dir),
        os.path.abspath(embedding_model_path),
        _embedding_device(),
    )
    if key not in _FAISS_BUNDLE_CACHE:
        embeddings = _build_embeddings(embedding_model_path)
        try:
            db = FAISS.load_local(
                faiss_dir,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except TypeError:
            db = FAISS.load_local(faiss_dir, embeddings)
        _maybe_move_faiss_index_to_gpu(db, device=0)
        _FAISS_BUNDLE_CACHE[key] = (embeddings, db)
    return _FAISS_BUNDLE_CACHE[key]


def _iter_faiss_documents(db: Any) -> List[Any]:
    """按 FAISS 行号顺序取出与向量矩阵对齐的 Document 列表。"""
    try:
        n = int(db.index.ntotal)
        mapping = db.index_to_docstore_id
        out: List[Any] = []
        for i in range(n):
            doc_id = mapping[i]
            out.append(db.docstore.search(doc_id))
        return out
    except Exception:
        store = getattr(db.docstore, "_dict", None)
        if isinstance(store, dict) and store:
            return list(store.values())
        raise


def _tokenize_for_bm25(text: str) -> List[str]:
    """中文 BM25：优先 jieba 分词；未安装时退化为字/英文片段。"""
    t = (text or "").strip()
    if not t:
        return []
    try:
        import jieba

        return [x for x in jieba.cut(t) if x.strip()]
    except ImportError:
        return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", t)


def _get_bm25_bundle(faiss_dir: str, embedding_model_path: str) -> Tuple[Any, List[Any]]:
    key = (
        os.path.abspath(faiss_dir),
        os.path.abspath(embedding_model_path),
        _embedding_device(),
    )
    if key not in _BM25_BUNDLE_CACHE:
        from rank_bm25 import BM25Okapi

        _, db = _get_faiss_bundle(faiss_dir, embedding_model_path)
        docs = _iter_faiss_documents(db)
        tokenized_corpus = [_tokenize_for_bm25(getattr(d, "page_content", "") or "") for d in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        _BM25_BUNDLE_CACHE[key] = (bm25, docs)
    return _BM25_BUNDLE_CACHE[key]


def retrieve_topk_bm25(
    query: str,
    faiss_dir: str = "./faiss_index",
    embedding_model_path: str = "./bge-m3",
    top_k: int = 20,
) -> List[Any]:
    """词法检索：与 FAISS 同一批 child 文档上的 BM25 Top-K。"""
    import numpy as np

    bm25, docs = _get_bm25_bundle(faiss_dir, embedding_model_path)
    q_tokens = _tokenize_for_bm25(query)
    if not q_tokens or not docs:
        return []
    scores = bm25.get_scores(q_tokens)
    idx = np.argsort(np.asarray(scores, dtype=np.float64))[::-1][:top_k]
    return [docs[int(i)] for i in idx]


def reciprocal_rank_fusion(
    ranked_lists: List[List[Any]],
    rrf_k: int = 60,
) -> List[Any]:
    """
    RRF：多路排序融合（常用 k=60）。输入为若干「已按相关性排序」的 Document 列表。
    """
    scores: Dict[str, float] = {}
    doc_by_key: Dict[str, Any] = {}
    for rank_list in ranked_lists:
        for rank, doc in enumerate(rank_list, start=1):
            key = _normalize_passage_for_dedup(getattr(doc, "page_content", "") or "")
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            if key not in doc_by_key:
                doc_by_key[key] = doc
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [doc_by_key[k] for k in sorted_keys]


def _hybrid_enabled() -> bool:
    v = (os.environ.get("RAG_USE_HYBRID") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def clear_search_caches() -> None:
    """释放缓存（测试或切换索引/嵌入设备前可调用）。"""
    _FAISS_BUNDLE_CACHE.clear()
    _RERANKER_CACHE.clear()
    _BM25_BUNDLE_CACHE.clear()


def retrieve_topk_from_faiss(
    query: str,
    faiss_dir: str = "./faiss_index",
    embedding_model_path: str = "./bge-m3",
    top_k: int = 20,
):
    _, db = _get_faiss_bundle(faiss_dir, embedding_model_path)
    # 返回的是 Document：包含 page_content 和 metadata
    return db.similarity_search(query, k=top_k)


def rerank_with_bge_reranker(
    query: str,
    candidates: List[object],
    reranker_model_path: str = "./bge-reranker-v2-m3",
    use_fp16: bool = True,
    normalize: bool = True,
):
    """
    candidates: 来自 FAISS 的 Document 列表（page_content + metadata）
    返回按分数降序排列的全部候选（由上层再按 final_k 去重截断）。
    """
    # FlagEmbedding 的用法见：bge-reranker-v2-m3/README.md
    from FlagEmbedding import FlagReranker

    rk = (os.path.abspath(reranker_model_path), use_fp16)
    if rk not in _RERANKER_CACHE:
        _RERANKER_CACHE[rk] = FlagReranker(reranker_model_path, use_fp16=use_fp16)
    reranker = _RERANKER_CACHE[rk]
    passages = [d.page_content for d in candidates]

    pairs: List[Tuple[str, str]] = [[query, p] for p in passages]
    scores = reranker.compute_score(pairs, normalize=normalize)

    ranked = []
    for doc, score in zip(candidates, scores):
        ranked.append(
            {
                "page_content": doc.page_content,
                "metadata": getattr(doc, "metadata", {}),
                "score": float(score),
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def search_with_rerank(
    query: str,
    top_k: int = 20,
    final_k: int = 5,
    faiss_dir: str = "./faiss_index",
    embedding_model_path: str = "./bge-m3",
    reranker_model_path: str = "./bge-reranker-v2-m3",
    use_hybrid: Optional[bool] = None,
    rrf_k: int = 60,
    hybrid_pool: int = 40,
    retrieval_mode: Optional[str] = "hybrid",
):
    """
    检索 + 重排。默认启用混合检索：FAISS 语义 Top-K + BM25 词法 Top-K，经 RRF 合并后再去重、交交叉编码器重排。

    use_hybrid: None 时读环境变量 RAG_USE_HYBRID（默认开启）；显式 True/False 覆盖环境。
    hybrid_pool: RRF 合并后截断条数，再送重排器（避免交叉注意力过长）。

    retrieval_mode: 显式 "dense" | "sparse" | "hybrid" 时强制该模式（用于消融，且忽略 use_hybrid）。
      - dense：仅 FAISS 语义检索 top_k → 去重 → 重排。
      - sparse：仅 BM25（取 max(hybrid_pool, top_k*2) 条）→ 去重 → 重排。
      - hybrid：语义 + BM25 + RRF（与 use_hybrid=True 一致）。
    """
    if retrieval_mode is not None:
        mode = retrieval_mode.strip().lower()
        if mode not in ("dense", "sparse", "hybrid"):
            raise ValueError(f"retrieval_mode 应为 'dense'、'sparse' 或 'hybrid'，收到: {retrieval_mode!r}")
    else:
        mode = "hybrid" if (_hybrid_enabled() if use_hybrid is None else bool(use_hybrid)) else "dense"

    candidates: List[Any]
    if mode == "dense":
        candidates = retrieve_topk_from_faiss(
            query=query,
            faiss_dir=faiss_dir,
            embedding_model_path=embedding_model_path,
            top_k=top_k,
        )
    elif mode == "sparse":
        try:
            candidates = retrieve_topk_bm25(
                query=query,
                faiss_dir=faiss_dir,
                embedding_model_path=embedding_model_path,
                top_k=max(hybrid_pool, top_k * 2),
            )
        except Exception as e:
            print(f"[warn] BM25-only retrieval failed: {e}")
            candidates = []
    else:
        dense = retrieve_topk_from_faiss(
            query=query,
            faiss_dir=faiss_dir,
            embedding_model_path=embedding_model_path,
            top_k=top_k,
        )
        try:
            sparse = retrieve_topk_bm25(
                query=query,
                faiss_dir=faiss_dir,
                embedding_model_path=embedding_model_path,
                top_k=top_k,
            )
            merged = reciprocal_rank_fusion([dense, sparse], rrf_k=rrf_k)
            candidates = merged[: max(hybrid_pool, top_k * 2)]
        except Exception as e:
            print(f"[warn] hybrid retrieval failed, dense only: {e}")
            candidates = dense

    candidates = dedupe_candidates_by_content(candidates)
    if not candidates:
        return []
    reranked = rerank_with_bge_reranker(
        query=query,
        candidates=candidates,
        reranker_model_path=reranker_model_path,
    )
    return dedupe_ranked_top_k(reranked, final_k)


if __name__ == "__main__":
    q = input("输入查询：").strip()
    results = search_with_rerank(q, top_k=20, final_k=5)
    print("\n重排后的 Top 结果：\n")
    for i, r in enumerate(results, 1):
        law = r["metadata"].get("Law_Article") or r["metadata"].get("Law_Section") or ""
        print(f"[{i}] score={r['score']:.4f} {law}")
        print(r["page_content"][:500].replace("\n", " "))
        print("-" * 80)

