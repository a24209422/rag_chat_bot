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
        _DOC_VECS = embed([d.text for d in DOCS], "RETRIEVAL_DOCUMENT")
    return _DOC_VECS


# ── 線上階段：問題也算成向量，找地圖上最近的鄰居 ──
# 門檻 0.84 是對 e5 量出來的（probe_threshold.py）。不能抄雲端的 0.70 ——
# e5 的分數整體偏高，連離題問題都有 0.79，0.70 在這裡等於沒有門檻。
# 間隙只有 0.046 寬（雲端 0.128），DOCS 一改動就要重量。
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
    for doc, score in retrieve("有哪些無人機相關的職缺？"):
        print(f"{score:.3f}  {doc.label}")
