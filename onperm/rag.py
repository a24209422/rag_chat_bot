# onperm/rag.py（地端版檢索）
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
# 知識庫兩邊共用——同一批 DOCS 才比得出雲端與地端的差異
from knowledge import DOCS   # noqa: E402
from facets import parse_query, match, known_districts   # noqa: E402
                                       # ↑ 精確過濾，見 shared/facets.py

# ── 離線階段：把每段話算成向量（放到「意思的地圖」上）──
# import 時不載模型也不建索引，第一次檢索才做（比照雲端版 cloud/rag.py）
_EMBEDDER = None
# 區名詞彙表：問句裡的「內湖」沒有「區」字可當錨點，只能靠詞彙表比對。
# 從 DOCS 長出來，所以重建 jobs.json 之後會自動跟著更新。
DISTRICTS = known_districts(DOCS)

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
def retrieve(question, k=5, min_score=0.82, filters=None):
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
        filters = parse_query(question, DISTRICTS)
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
    for doc, score in retrieve("有哪些無人機相關的職缺？"):
        print(f"{score:.3f}  {doc.label}")
