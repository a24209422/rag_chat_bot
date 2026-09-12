# onperm/rag.py（地端版檢索）
from functools import lru_cache

from shared.retriever import BaseRetriever

MODEL = "intfloat/multilingual-e5-small"

# e5 是用 "query: " / "passage: " 前綴區分查詢與文件，Gemini 是用 task_type 參數。
# 這裡把 task_type 映成前綴，函式形狀就跟 cloud/rag.py 的 embed 一樣。
_PREFIX = {"RETRIEVAL_QUERY": "query: ", "RETRIEVAL_DOCUMENT": "passage: "}


@lru_cache(maxsize=2)
def load_embedder(name=MODEL):
    """載入 SentenceTransformer，依模型名快取。

    快取是因為載入很貴（第一次還要下載約 470MB），不是因為需要全域狀態——
    所以做成「帶參數的函式 + lru_cache」而不是模組層的 _EMBEDDER：
    要比較兩個 embedding 模型時各拿各的，不會互相覆蓋。

    import 也擺在函式裡：sentence_transformers 光是 import 就要 25 秒
    （實測），而 import onperm.rag 的人不一定要用到模型——測試、
    probe_threshold 挑另一邊、providers 只是被載入，都不該付這個成本。
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


class OnpremRetriever(BaseRetriever):
    # 0.82 只當「下限」用，不是離題守門員——這點跟雲端的 0.65 不同。
    # 對這批資料量出來的間隙只有 0.003 寬（FAQ 時代是 0.046）：離題問句
    # 「推薦一家餐廳」0.863 跟正常問句「耐能智慧在徵什麼人」0.866 幾乎貼在一起。
    # 原因是統計性的，不是門檻選錯：候選塊從 5 個變成 140 個，雜訊的最大值
    # 被推高，但正確答案的分數不會跟著漲。雲端 Gemini 的間隙有 0.092 寬，
    # 所以那側的門檻擋得住離題，這側擋不住。
    # 離題的判斷改由模型負責（見 shared/knowledge.py 的 SYSTEM）。
    # 0.82 = 有的最低 0.866 減 0.05，作用是擋掉明顯無關的塊，不是擋問句。
    # 重建 jobs.json 之後要重跑：python -m tools.probe_threshold onperm
    min_score = 0.82

    def __init__(self, docs=None, min_score=None, model=MODEL):
        super().__init__(docs=docs, min_score=min_score)
        self.model = model

    @property
    def embedder(self):
        return load_embedder(self.model)     # 第一次要下載模型，會慢一下

    def embed(self, texts, task_type):
        prefix = _PREFIX[task_type]      # 打錯字就當場 KeyError，不要默默算出爛向量
        return self.embedder.encode([prefix + t for t in texts],
                                    normalize_embeddings=True)


if __name__ == "__main__":       # python -m onperm.rag
    import sys
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 主控台預設 cp950，中文會亂碼
    for doc, score in OnpremRetriever().retrieve("有哪些無人機相關的職缺？"):
        print(f"{score:.3f}  {doc.label}")
