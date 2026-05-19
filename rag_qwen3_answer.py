import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_search_rerank import search_with_rerank


# 未设置环境变量时默认暴露物理 GPU 0。若机器只有一块卡却写 "1" 会导致看不到 GPU、推理与检索全落 CPU。
# 多卡且要用另一张卡时：启动前 export CUDA_VISIBLE_DEVICES=1（或所需编号）。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# 完全相同问题复用上次答案（LRU，避免无限增长）
_QUERY_ANSWER_CACHE: "OrderedDict[str, str]" = OrderedDict()
_QUERY_ANSWER_CACHE_MAX = 128


def _query_cache_key(query: str) -> str:
    return query.strip()


def get_cached_answer(query: str) -> Optional[str]:
    key = _query_cache_key(query)
    if not key:
        return None
    if key not in _QUERY_ANSWER_CACHE:
        return None
    _QUERY_ANSWER_CACHE.move_to_end(key)
    return _QUERY_ANSWER_CACHE[key]


def set_cached_answer(query: str, answer: str) -> None:
    key = _query_cache_key(query)
    if not key:
        return
    if key in _QUERY_ANSWER_CACHE:
        del _QUERY_ANSWER_CACHE[key]
    _QUERY_ANSWER_CACHE[key] = answer
    _QUERY_ANSWER_CACHE.move_to_end(key)
    while len(_QUERY_ANSWER_CACHE) > _QUERY_ANSWER_CACHE_MAX:
        _QUERY_ANSWER_CACHE.popitem(last=False)


def clear_query_answer_cache() -> None:
    _QUERY_ANSWER_CACHE.clear()


def build_context_block(
    query: str,
    ranked_results: List[Dict],
    tokenizer,
    max_context_tokens: int = 3000,
) -> str:
    """
    将重排后的前几条候选拼成一个 Context 块，并尽量不超过 max_context_tokens。
    """
    context_parts: List[str] = []
    used_tokens = 0

    seen_parent: set[str] = set()
    slot = 0
    for r in ranked_results:
        meta = r.get("metadata") or {}
        pid = (meta.get("parent_id") or "").strip()
        if pid and pid in seen_parent:
            continue
        if pid:
            seen_parent.add(pid)

        slot += 1
        law = meta.get("Law_Article") or meta.get("Law_Section") or meta.get("Law_Chunk") or ""
        passage = (meta.get("parent_content") or r.get("page_content") or "").strip()

        item = f"[{slot}] {law}\n{passage}\n"
        item_tokens = len(tokenizer.encode(item, add_special_tokens=False))
        if used_tokens + item_tokens > max_context_tokens:
            break
        context_parts.append(item)
        used_tokens += item_tokens

    context_text = "\n".join(context_parts).strip()
    if not context_text:
        context_text = "（无有效检索材料）"
    return context_text


def load_answer_model(
    model_path: str = "./Qwen3-8B",
    use_fp16: bool = True,
) -> Tuple[Any, Any]:
    """
    加载问答用 Qwen（只应在进程内调用一次；批量填 JSONL 时复用同一实例）。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if not use_fp16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def strip_thinking(text: str) -> str:
    """
    尝试去掉 <think>...</think> 内容，保证只输出最终回答。
    """
    if "<think>" in text and "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def generate_answer(
    query: str,
    model_path: str = "./Qwen3-8B",
    top_k: int = 20,
    final_k_for_llm: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.95,
    use_fp16: bool = True,
) -> str:
    cached = get_cached_answer(query)
    if cached is not None:
        return cached

    ranked = search_with_rerank(
        query=query,
        top_k=top_k,
        final_k=final_k_for_llm,
    )

    ans = answer_from_ranked(
        query=query,
        ranked_results=ranked,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_fp16=use_fp16,
    )
    set_cached_answer(query, ans)
    return ans


def answer_from_ranked(
    query: str,
    ranked_results: List[Dict],
    model_path: str = "./Qwen3-8B",
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.95,
    use_fp16: bool = True,
    *,
    tokenizer: Optional[Any] = None,
    model: Optional[Any] = None,
) -> str:
    if tokenizer is None or model is None:
        tokenizer, model = load_answer_model(model_path, use_fp16=use_fp16)

    context = build_context_block(
        query=query,
        ranked_results=ranked_results,
        tokenizer=tokenizer,
        max_context_tokens=3000,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "你是法律问答助手。请严格根据“材料”回答问题。"
                "如果材料中没有直接依据，请说“无法从提供材料判断”。"
                "回答时尽量引用材料编号（例如 [1] [2]）。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"问题：{query}\n\n"
                f"材料：\n{context}\n\n"
                "请给出结论，并简要说明依据。"
            ),
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # 你也可以改 True，但输出会包含 <think>...</think>
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    # 只取新增部分
    new_tokens = generated_ids[0][inputs.input_ids.shape[-1] :].tolist()
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return strip_thinking(output_text)


if __name__ == "__main__":
    print("RAG 问答模式：输入问题后回车。输入 `q/exit/quit` 退出。")
    while True:
        q = input("\n输入问题：").strip()
        if not q:
            continue
        if q.lower() in {"q", "exit", "quit"}:
            break

        t0 = time.perf_counter()

        cached = get_cached_answer(q)
        if cached is not None:
            t1 = time.perf_counter()
            print("\n（本问题与缓存完全一致，跳过检索与生成）\n")
            print("\n--- 答案 ---\n")
            print(cached)
            print(f"\n本次总耗时：{t1 - t0:.2f}s")
            continue

        ranked = search_with_rerank(q, top_k=20, final_k=5)

        print("\n--- 重排后 Top 材料（已去重） ---\n")
        for i, r in enumerate(ranked, 1):
            meta = r.get("metadata") or {}
            law = meta.get("Law_Article") or meta.get("Law_Section") or meta.get("Law_Chunk") or ""
            print(f"[{i}] score={r['score']:.4f} {law}")
            print((r["page_content"] or "").strip()[:300].replace("\n", " "))
            print("-" * 80)

        ans = answer_from_ranked(q, ranked)
        set_cached_answer(q, ans)
        t1 = time.perf_counter()

        print("\n--- 答案 ---\n")
        print(ans)
        print(f"\n本次总耗时：{t1 - t0:.2f}s")

