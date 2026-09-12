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


def retrieve(question, k=5, min_score=0.82):
    """回傳 k 個「不同職缺」，不是 k 個塊。

    一個職缺被切成好幾塊，同一個問句常常同時撈到同一職缺的多塊
    （實測 12 個問句有 9 個發生），不去重的話同一份 full 會被重複塞進
    prompt，既浪費 context 又排擠掉其他職缺。

    min_score 現在只是個下限，用來擋掉明顯無關的塊——它已經不能當
    「離題守門員」了：對這批資料量出來的間隙只有 0.003 寬
    （FAQ 時代是 0.046），離題問句「推薦一家餐廳」0.863 跟正常問句
    「耐能智慧在徵什麼人」0.866 幾乎貼在一起。原因是候選塊從 5 個變成
    140 個，雜訊的最大值被推高，但正確答案的分數不會跟著漲。
    離題的判斷改由模型負責（見 shared/knowledge.py 的 SYSTEM）。
    """
    qv = embed([question], "RETRIEVAL_QUERY")[0]
    scores = doc_vecs() @ qv
    out, seen = [], set()
    for i in np.argsort(-scores):              # 由高到低掃
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