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
from facets import parse_query, match   # noqa: E402 ← 精確過濾，見 shared/facets.py

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


def retrieve(question, k=5, min_score=0.70, filters=None):
    """回傳 k 個「不同職缺」，不是 k 個塊。

    先看問句有沒有可以精確過濾的條件（地點、工作性質、學歷），有的話就
    先篩再排序。這是為了「篩選型」問題——問「有哪些台北的職缺」時，純語意
    相似度只會給你最像的前 k 個，沒辦法保證「全部」；但地點本來就是結構化
    欄位，直接篩就能保證完整。見 shared/facets.py。

    有過濾條件時會放掉 min_score 和 k：篩選已經保證相關，分數門檻反而會
    誤殺；k 也要放開，不然問「有哪些台北的職缺」只給 5 個（實際有 20 個）。

    沒有過濾條件時就是純向量檢索。min_score 只當下限用，擋掉明顯無關的塊
    ——它已經不能當離題守門員了：對這批資料量出來的間隙只有 0.003 寬
    （FAQ 時代是 0.046），離題問句「推薦一家餐廳」0.863 跟正常問句
    「耐能智慧在徵什麼人」0.866 幾乎貼在一起。原因是候選塊從 5 個變成
    140 個，雜訊的最大值被推高，但正確答案的分數不會跟著漲。
    離題的判斷改由模型負責（見 shared/knowledge.py 的 SYSTEM）。

    同職缺去重：一個職缺切成好幾塊，實測 12 個問句有 9 個會同時撈到同職缺
    的多塊，不去重的話同一份 full 會被重複塞進 prompt。
    """
    if filters is None:
        filters = parse_query(question)
    qv = embed([question], "RETRIEVAL_QUERY")[0]
    scores = doc_vecs() @ qv

    if filters:
        cand = [i for i in range(len(DOCS)) if match(DOCS[i].facets, filters)]
        min_score = 0.0                                   # 篩過了，不再用分數擋
        k = max(k, len({DOCS[i].group for i in cand}))     # 要給全部，不是前 k 個
        order = sorted(cand, key=lambda i: -scores[i])
    else:
        order = np.argsort(-scores)

    out, seen = [], set()
    for i in order:
        if scores[i] < min_score:
            break                              # 已排序，低於門檻後面不用看了
        d = DOCS[i]
        if d.group in seen:                    # 同職缺只留最高分那塊
            continue
        seen.add(d.group)
        out.append((d, float(scores[i])))
        if len(out) == k:
            break
    return out


if __name__ == "__main__":
    question = "有哪些無人機相關的職缺？"
    for doc, score in retrieve(question):
        print(f"{score:.4f}  {doc.label}")