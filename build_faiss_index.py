"""
FAISS 建库：结构对齐（Parent = 每条 ### 法条行）+ BGE-M3 句间相似度语义切分（Child）+ 父子元数据。

- 阶段 A：每行 `### [法律key] 正文` 为一个 Parent，不跨行合并。
- 阶段 B：在 Parent 内按中文句读切句，用 BGE-M3 算相邻句余弦相似度，在相似度低谷处切段；过长块再按字符上限截断。
- 向量库仅嵌入 Child；metadata 含 parent_id、parent_content（整行 Parent），供生成时展开全文。

依赖：numpy（与 torch/scikit 环境通常已有）
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

MODEL_PATH = "./bge-m3"
MANIFEST_FILE = "manifest.json"

# 语义切分：相邻句相似度低于阈值则断块（结合均值−k·标准差自适应）
SEMANTIC_SIM_BASE = 0.35
SEMANTIC_SIM_STD_COEF = 0.35
SEMANTIC_SIM_MIN = 0.30
SEMANTIC_SIM_MAX = 0.58

# Child 长度：过小则合并，过大则硬切
MAX_CHILD_CHARS = 2500
MIN_CHILD_CHARS = 80
HARD_SPLIT_OVER_CHARS = 2200

H3_LINE = re.compile(r"^### \[([^\]]+)\]\s*(.+?)\s*$")

device_str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在将 BGE-M3 加载到设备: {device_str}")
embeddings = HuggingFaceBgeEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs={"device": device_str},
    encode_kwargs={"normalize_embeddings": True},
)


def _split_sentences_zh(text: str) -> List[str]:
    """按中文句末标点切句；无标点长串则整块返回。"""
    t = (text or "").strip()
    if not t:
        return []
    segs = re.findall(r"[^。；！？\n]+[。；！？]?", t)
    out = [s.strip() for s in segs if s.strip()]
    if not out:
        return [t]
    return out


def _hard_wrap(text: str, limit: int, step: int) -> List[str]:
    if len(text) <= limit:
        return [text] if text.strip() else []
    return [text[i : i + limit] for i in range(0, len(text), step) if text[i : i + limit].strip()]


def _merge_and_cap_chunks(chunks: List[str]) -> List[str]:
    """合并过短子块，并对过长块硬切。"""
    if not chunks:
        return []
    merged: List[str] = []
    buf = ""
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if not buf:
            buf = c
            continue
        if len(buf) < MIN_CHILD_CHARS or len(c) < MIN_CHILD_CHARS:
            buf = buf + c
        elif len(buf) + len(c) <= MAX_CHILD_CHARS:
            buf = buf + c
        else:
            merged.extend(_hard_wrap(buf, MAX_CHILD_CHARS, MAX_CHILD_CHARS - 200))
            buf = c
    if buf:
        merged.extend(_hard_wrap(buf, MAX_CHILD_CHARS, MAX_CHILD_CHARS - 200))
    return [x for x in merged if x.strip()]


def semantic_chunk_parent_text(parent_line: str) -> List[str]:
    """
    在单条 Parent 正文内做 BGE-M3 语义切分，返回 Child 文本列表（不含重复 ### 行头时可仅为正文切片）。
    """
    parent_line = (parent_line or "").strip()
    if not parent_line:
        return []

    if len(parent_line) <= MIN_CHILD_CHARS:
        return [parent_line]

    sentences = _split_sentences_zh(parent_line)
    if len(sentences) <= 1:
        return _merge_and_cap_chunks(_hard_wrap(parent_line, HARD_SPLIT_OVER_CHARS, HARD_SPLIT_OVER_CHARS - 200))

    embs = embeddings.embed_documents(sentences)
    arr = np.asarray(embs, dtype=np.float64)
    sims = np.sum(arr[:-1] * arr[1:], axis=1)

    if len(sims) == 0:
        return [parent_line]

    mean_s = float(np.mean(sims))
    std_s = float(np.std(sims)) if len(sims) > 1 else 0.0
    thresh = mean_s - SEMANTIC_SIM_STD_COEF * std_s
    thresh = float(np.clip(thresh, SEMANTIC_SIM_MIN, SEMANTIC_SIM_MAX))
    thresh = max(thresh, SEMANTIC_SIM_BASE)

    breaks: List[int] = [0]
    for i in range(len(sims)):
        if sims[i] < thresh:
            breaks.append(i + 1)
    if breaks[-1] != len(sentences):
        breaks.append(len(sentences))

    raw_chunks: List[str] = []
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        piece = "".join(sentences[lo:hi]).strip()
        if piece:
            raw_chunks.append(piece)

    if not raw_chunks:
        raw_chunks = [parent_line]

    return _merge_and_cap_chunks(raw_chunks)


def _parse_parent_lines(content: str) -> List[Tuple[str, str]]:
    """从 markdown 抽出 (law_key, 整行含 ###)。"""
    out: List[Tuple[str, str]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        m = H3_LINE.match(line)
        if m:
            out.append((m.group(1).strip(), line))
    return out


def split_md_to_parent_child_documents(
    md_file: Path,
    content: str,
) -> List[Document]:
    """
    阶段 A：每行 ### 为 Parent；阶段 B：Parent 内语义切分为 Child。
    写入 FAISS 的为 Child；metadata.parent_content 为完整 Parent 行。
    """
    parents = _parse_parent_lines(content)
    docs: List[Document] = []

    for pidx, (law_key, parent_line) in enumerate(parents):
        body_for_header = ""
        m = H3_LINE.match(parent_line)
        if m:
            body_for_header = m.group(2).strip()
        law_display = body_for_header[:220] if body_for_header else law_key

        parent_id = f"{md_file.stem}::p{pidx}"

        children_text = semantic_chunk_parent_text(parent_line)
        if not children_text:
            continue

        for cidx, child_text in enumerate(children_text):
            meta = {
                "Law_Article": law_display,
                "law_key": law_key,
                "parent_id": parent_id,
                "parent_content": parent_line,
                "chunk_role": "child",
                "child_idx": cidx,
                "struct_idx": pidx,
            }
            docs.append(
                Document(
                    page_content=child_text,
                    metadata=meta,
                )
            )

    return docs


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(save_dir: str) -> Path:
    return Path(save_dir) / MANIFEST_FILE


def _load_manifest(save_dir: str) -> Dict[str, str]:
    p = _manifest_path(save_dir)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(save_dir: str, manifest: Dict[str, str]) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(_manifest_path(save_dir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _load_md_to_chunked_docs(md_file: Path) -> List[Document]:
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()
    chunked = split_md_to_parent_child_documents(md_file, content)
    for doc in chunked:
        meta = dict(doc.metadata or {})
        meta["source_file"] = md_file.name
        doc.metadata = meta
    return chunked


def _build_index_from_md_files(md_files: List[Path], save_dir: str) -> int:
    all_docs: List[Document] = []
    for p in md_files:
        all_docs.extend(_load_md_to_chunked_docs(p))
    if not all_docs:
        return 0
    db = FAISS.from_documents(all_docs, embeddings)
    db.save_local(save_dir)
    return len(all_docs)


def incremental_index_markdown_files(md_files: List[str], save_dir: str = "./faiss_index") -> Dict[str, int]:
    """
    对指定 markdown 文件做增量索引：
    - 新文件/内容变化：写入向量库
    - 未变化：跳过
    """
    save_path = Path(save_dir)
    manifest = _load_manifest(save_dir)

    changed_files: List[Path] = []
    need_rebuild = False
    for f in md_files:
        p = Path(f)
        if not p.exists() or p.suffix.lower() != ".md":
            continue
        digest = _sha256_of_file(p)
        old_digest = manifest.get(p.name)
        if old_digest == digest:
            continue
        if old_digest is not None and old_digest != digest:
            need_rebuild = True
        changed_files.append(p)

    if not changed_files:
        return {"indexed_files": 0, "indexed_chunks": 0}

    indexed_chunks = 0
    if need_rebuild:
        all_md = [Path(f) for f in md_files if Path(f).exists() and Path(f).suffix.lower() == ".md"]
        indexed_chunks = _build_index_from_md_files(all_md, save_dir)
        print("检测到已收录文件内容变更，已执行一次全量重建以避免旧向量残留。")
    else:
        new_docs: List[Document] = []
        for p in changed_files:
            print(f"增量处理: {p.name}")
            new_docs.extend(_load_md_to_chunked_docs(p))
        if not new_docs:
            return {"indexed_files": 0, "indexed_chunks": 0}
        if save_path.exists() and any(save_path.iterdir()):
            try:
                db = FAISS.load_local(save_dir, embeddings, allow_dangerous_deserialization=True)
            except TypeError:
                db = FAISS.load_local(save_dir, embeddings)
            db.add_documents(new_docs)
        else:
            db = FAISS.from_documents(new_docs, embeddings)
        db.save_local(save_dir)
        indexed_chunks = len(new_docs)

    for p in changed_files:
        manifest[p.name] = _sha256_of_file(p)
    _save_manifest(save_dir, manifest)

    return {"indexed_files": len(changed_files), "indexed_chunks": indexed_chunks}


def run_indexing(md_dir: str = "./markdown", save_dir: str = "./faiss_index") -> None:
    """
    扫描 markdown 目录并执行增量索引。
    """
    md_path = Path(md_dir)
    if not md_path.exists():
        print(f"错误：找不到目录 {md_path.resolve()}")
        return
    md_files = [str(p) for p in md_path.iterdir() if p.suffix.lower() == ".md"]
    stats = incremental_index_markdown_files(md_files, save_dir=save_dir)
    print(
        f"增量索引完成：更新文件 {stats['indexed_files']} 个，新增片段 {stats['indexed_chunks']} 条；"
        f"索引目录：{Path(save_dir).resolve()}"
    )


if __name__ == "__main__":
    run_indexing()
