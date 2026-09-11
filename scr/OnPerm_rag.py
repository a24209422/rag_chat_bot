# mini_rag.py
import numpy as np
from sentence_transformers import SentenceTransformer

# ── 你的「知識庫」。最簡單的形式就是一個 list，一段一句話 ──
DOCS = [
    "退款流程：商品收到後 7 天內可申請退款，需保持包裝完整。",
    "運費說明：訂單滿 1000 元免運，未滿加收 80 元。",
    "客服時間：週一至週五 09:00-18:00，例假日不營業。",
    "會員等級：累積消費滿 5000 元升級為金卡，享 95 折。",
    "保固政策：電子產品保固一年，人為損壞不在範圍內。",
]

# ── 離線階段：把每段話算成向量（放到「意思的地圖」上）──
# import 時不載模型也不建索引，第一次檢索才做（比照雲端版 cloud_rag.py）
_EMBEDDER = None
_DOC_VECS = None


def embedder():
    global _EMBEDDER
    if _EMBEDDER is None:                  # 第一次要下載模型，會慢一下
        _EMBEDDER = SentenceTransformer("intfloat/multilingual-e5-small")
    return _EMBEDDER


# e5 是用 "query: " / "passage: " 前綴區分查詢與文件，Gemini 是用 task_type 參數。
# 這裡把 task_type 映成前綴，函式形狀就跟 cloud_rag.embed 一樣 ——
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
