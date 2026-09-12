# cloud/rag.py（Gemini 版檢索）
import re
import sys
import time

import numpy as np
from google.genai import errors, types

from cloud.client import make_client
from shared.retriever import BaseRetriever
from shared.settings import settings

# Gemini 的兩個限制，兩個都是 FAQ 時代（5 筆）不會碰到、換成職缺（140 塊）才炸出來的：
#   1. BatchEmbedContentsRequest 一次最多 100 筆 → 400 INVALID_ARGUMENT（見 batch）
#   2. 免費方案「每分鐘 100 個 request」，而 batch 裡每段文字各算一個 → 429


class CloudRetriever(BaseRetriever):
    # 0.65 是對職缺資料量出來的（python -m tools.probe_threshold cloud）。
    # 原本的 0.70 是 FAQ 時代的值，換成職缺後會擋掉「有沒有可以全遠端的
    # 工作？」——它只有 0.695。Gemini 的間隙有 0.092 寬（地端 e5 只有
    # 0.003），所以雲端這側門檻還真的擋得住離題，不像地端得交給模型判斷。
    min_score = 0.65

    def __init__(self, docs=None, min_score=None, client=None, config=None,
                 store_path=None, registry=None):
        """client 不傳就自己建一個（延遲到第一次用才讀金鑰）。

        store_path 不傳就用設定裡的路徑；傳 False 代表不落地（測試用）。
        """
        cfg = config or settings()
        super().__init__(
            docs=docs, min_score=min_score, registry=registry,
            store_path=cfg.cloud_store_path if store_path is None else store_path or None)
        self._cfg = cfg
        self._client = client
        self.model = cfg.gemini_embed_model
        self.embed_model = cfg.gemini_embed_model
        self.dim = cfg.gemini_embed_dim
        self.batch = cfg.gemini_embed_batch

    @property
    def client(self):
        if self._client is None:
            self._client = make_client(config=self._cfg)
        return self._client

    def embed(self, texts, task_type):
        vecs = []
        for i in range(0, len(texts), self.batch):
            part = texts[i:i + self.batch]
            for attempt in range(6):
                try:
                    resp = self.client.models.embed_content(
                        model=self.model,
                        contents=part,
                        config=types.EmbedContentConfig(
                            task_type=task_type,   # ← 取代 e5 的 "passage:" / "query:" 前綴
                            output_dimensionality=self.dim,
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


if __name__ == "__main__":       # python -m cloud.rag
    import sys
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 主控台預設 cp950，中文會亂碼
    question = "有哪些無人機相關的職缺？"
    for doc, score in CloudRetriever().retrieve(question):
        print(f"{score:.4f}  {doc.label}")
