# cloud/rag.py（Gemini 版檢索）
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from knowledge import DOCS   # noqa: E402 ← 兩邊共用，見 shared/knowledge.py

load_dotenv()

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
        _DOC_VECS = embed([d.text for d in DOCS], "RETRIEVAL_DOCUMENT")   # ← 從 import 時搬到這裡
    return _DOC_VECS


def retrieve(question, k=2, min_score=0.70):       # ← 門檻用量的，見 probe_threshold.py
    qv = embed([question], "RETRIEVAL_QUERY")[0]    # 線上查詢
    scores = doc_vecs() @ qv
    top = np.argsort(-scores)[:k]
    return [(DOCS[i], float(scores[i])) for i in top if scores[i] >= min_score]


if __name__ == "__main__":
    question = "如何申請退款？"
    for doc, score in retrieve(question):
        print(f"{score:.4f}  {doc.label}")