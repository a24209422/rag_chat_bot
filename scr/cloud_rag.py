# cloud_rag.py（Gemini 版）
import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DOCS = [
    "退款流程：商品收到後 7 天內可申請退款，需保持包裝完整。",
    "運費說明：訂單滿 1000 元免運，未滿加收 80 元。",
    "客服時間：週一至週五 09:00-18:00，例假日不營業。",
    "會員等級：累積消費滿 5000 元升級為金卡，享 95 折。",
    "保固政策：電子產品保固一年，人為損壞不在範圍內。",
]

KEY = os.environ.get("GEMINI_API_KEY")          # ← 不要用 os.environ[...]
if not KEY:
    raise RuntimeError("找不到 GEMINI_API_KEY：本機放 .env，雲端放 App settings → Secrets")
client = genai.Client(api_key=KEY)
MODEL = "gemini-embedding-001"
DIM = 768        # 預設 3072；截短成 768 省記憶體與比對時間（靠 [[MRL]]）


def embed(texts, task_type):
    resp = client.models.embed_content(
        model=MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,               # ← 取代 e5 的 "passage:" / "query:" 前綴
            output_dimensionality=DIM,
        ),
    )
    v = np.array([e.values for e in resp.embeddings], dtype="float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)   # ← 非 3072 維時「必須」自己正規化
    return v


_DOC_VECS = None            # import 時不打 API，第一次檢索才建索引


def doc_vecs():
    global _DOC_VECS
    if _DOC_VECS is None:                           # 算過就重用
        _DOC_VECS = embed(DOCS, "RETRIEVAL_DOCUMENT")   # ← 從 import 時搬到這裡
    return _DOC_VECS


def retrieve(question, k=2, min_score=0.70):       # ← 門檻用量的，見 probe_threshold.py
    qv = embed([question], "RETRIEVAL_QUERY")[0]    # 線上查詢
    scores = doc_vecs() @ qv
    top = np.argsort(-scores)[:k]
    return [(DOCS[i], float(scores[i])) for i in top if scores[i] >= min_score]


if __name__ == "__main__":
    question = "如何申請退款？"
    for text, score in retrieve(question):
        print(f"{score:.4f}  {text}")