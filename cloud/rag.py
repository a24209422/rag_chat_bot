# cloud/rag.py（Gemini 版檢索）
import hashlib
import re
import sys
import time
from pathlib import Path

import numpy as np
from google.genai import errors, types

from cloud.client import make_client
from shared.retriever import BaseRetriever

MODEL = "gemini-embedding-001"
DIM = 768        # 預設 3072；截短成 768 省記憶體與比對時間（靠 [[MRL]]）

# Gemini 的兩個限制，兩個都是 FAQ 時代（5 筆）不會碰到、換成職缺（140 塊）才炸出來的：
#   1. BatchEmbedContentsRequest 一次最多 100 筆 → 400 INVALID_ARGUMENT
#   2. 免費方案「每分鐘 100 個 request」，而 batch 裡每段文字各算一個 → 429
BATCH = 100
CACHE = Path(__file__).resolve().parent.parent / "data" / "vecs_cloud.npz"


class CloudRetriever(BaseRetriever):
    # 0.65 是對職缺資料量出來的（python -m tools.probe_threshold cloud）。
    # 原本的 0.70 是 FAQ 時代的值，換成職缺後會擋掉「有沒有可以全遠端的
    # 工作？」——它只有 0.695。Gemini 的間隙有 0.092 寬（地端 e5 只有
    # 0.003），所以雲端這側門檻還真的擋得住離題，不像地端得交給模型判斷。
    min_score = 0.65

    def __init__(self, docs=None, min_score=None, client=None, cache=CACHE):
        """client 不傳就自己建一個（會讀 .env）。cache=None 可關掉磁碟快取。"""
        super().__init__(docs=docs, min_score=min_score)
        self._client = client
        self.cache = cache

    @property
    def client(self):
        if self._client is None:
            self._client = make_client()
        return self._client

    def embed(self, texts, task_type):
        vecs = []
        for i in range(0, len(texts), BATCH):
            part = texts[i:i + BATCH]
            for attempt in range(6):
                try:
                    resp = self.client.models.embed_content(
                        model=MODEL,
                        contents=part,
                        config=types.EmbedContentConfig(
                            task_type=task_type,   # ← 取代 e5 的 "passage:" / "query:" 前綴
                            output_dimensionality=DIM,
                        ),
                    )
                    break
                except errors.ClientError as e:
                    if e.code != 429 or attempt == 5:
                        raise
                    # 每分鐘配額用完了。錯誤訊息裡有「Please retry in 51.667s」，
                    # 直接從字串讀比翻 e.details 的結構穩（那裡面不是統一的 dict）。
                    m = re.search(r"retry in ([\d.]+)s", str(e))
                    wait = int(float(m.group(1))) + 5 if m else 60
                    print("  配額用完，等 %d 秒後重試（第 %d 次）" % (wait, attempt + 1),
                          file=sys.stderr)
                    time.sleep(wait)
            vecs += [e.values for e in resp.embeddings]
        v = np.array(vecs, dtype="float32")
        v /= np.linalg.norm(v, axis=1, keepdims=True)   # ← 非 3072 維時「必須」自己正規化
        return v

    def _build_doc_vecs(self):
        """建索引，並存成快取。

        地端重算只是慢十秒，雲端重算是燒 140 個 request 配額外加兩分鐘等待
        （免費方案每分鐘只給 100 個），所以雲端這側非快取不可。
        快取用 docs 的 text 算雜湊當 key——重建 jobs.json 之後會自動失效重算。
        """
        texts = [d.text for d in self.docs]
        if self.cache is None:
            return self.embed(texts, "RETRIEVAL_DOCUMENT")

        key = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
        if self.cache.exists():
            z = np.load(self.cache, allow_pickle=False)
            if str(z["key"]) == key:
                return z["vecs"]

        vecs = self.embed(texts, "RETRIEVAL_DOCUMENT")
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.cache, key=np.array(key), vecs=vecs)
        return vecs


if __name__ == "__main__":       # python -m cloud.rag
    import sys
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 主控台預設 cp950，中文會亂碼
    question = "有哪些無人機相關的職缺？"
    for doc, score in CloudRetriever().retrieve(question):
        print(f"{score:.4f}  {doc.label}")
