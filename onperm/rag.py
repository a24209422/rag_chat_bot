# onperm/rag.py（地端版檢索）
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
# 知識庫兩邊共用——同一批 DOCS 才比得出雲端與地端的差異
from knowledge import DOCS   # noqa: E402

# ── 離線階段：把每段話算成向量（放到「意思的地圖」上）──
# import 時不載模型也不建索引，第一次檢索才做（比照雲端版 cloud/rag.py）
_EMBEDDER = None
_DOC_VECS = None


def embedder():
    global _EMBEDDER
    if _EMBEDDER is None:                  # 第一次要下載模型，會慢一下
        _EMBEDDER = SentenceTransformer("intfloat/multilingual-e5-small")
    return _EMBEDDER


# e5 是用 "query: " / "passage: " 前綴區分查詢與文件，Gemini 是用 task_type 參數。
# 這裡把 task_type 映成前綴，函式形狀就跟 cloud/rag.py 的 embed 一樣 ——
# probe_threshold.py 這類工具只要換 import 就能對地端重量一次門檻。
_PREFIX = {"RETRIEVAL_QUERY": "query: ", "RETRIEVAL_DOCUMENT": "passage: "}


def embed(texts, task_type):
    prefix = _PREFIX[task_type]            # 打錯字就當場 KeyError，不要默默算出爛向量
    return embedder().encode([prefix + t for t in texts],
                             normalize_embeddings=True)


def doc_vecs():
    global _DOC_VECS
    if _DOC_VECS is None:                  # 算過就重用
        _DOC_VECS = embed(DOCS, "RETRIEVAL_DOCUMENT")
    return _DOC_VECS


# ── 線上階段：問題也算成向量，找地圖上最近的鄰居 ──
# 門檻 0.84 是對 e5 量出來的（probe_threshold.py）。不能抄雲端的 0.70 ——
# e5 的分數整體偏高，連離題問題都有 0.79，0.70 在這裡等於沒有門檻。
# 間隙只有 0.046 寬（雲端 0.128），DOCS 一改動就要重量。
def retrieve(question, k=2, min_score=0.84):
    qv = embed([question], "RETRIEVAL_QUERY")[0]
    scores = doc_vecs() @ qv               # 向量已正規化 → 內積就是餘弦相似度
    top = np.argsort(-scores)[:k]          # 由大到小排，取前 k 個
    return [(DOCS[i], float(scores[i])) for i in top if scores[i] >= min_score]


if __name__ == "__main__":
    for doc, score in retrieve("有什麼優惠"):
        print(f"{score:.3f}  {doc}")
