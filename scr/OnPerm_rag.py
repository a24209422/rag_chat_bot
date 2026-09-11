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


def doc_vecs():
    global _DOC_VECS
    if _DOC_VECS is None:                  # 算過就重用
        _DOC_VECS = embedder().encode(["passage: " + d for d in DOCS],
                                      normalize_embeddings=True)
    return _DOC_VECS


# ── 線上階段：問題也算成向量，找地圖上最近的鄰居 ──
def retrieve(question, k=2):
    qv = embedder().encode(["query: " + question], normalize_embeddings=True)[0]
    scores = doc_vecs() @ qv               # 向量已正規化 → 內積就是餘弦相似度
    top = np.argsort(-scores)[:k]          # 由大到小排，取前 k 個
    return [(DOCS[i], float(scores[i])) for i in top]


if __name__ == "__main__":
    for doc, score in retrieve("有什麼優惠"):
        print(f"{score:.3f}  {doc}")