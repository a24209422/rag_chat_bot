# cloud/rag.py（Gemini 版檢索）
import hashlib
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from knowledge import DOCS   # noqa: E402 ← 兩邊共用，見 shared/knowledge.py
from facets import parse_query, match, known_districts   # noqa: E402
                                       # ↑ 精確過濾，見 shared/facets.py

load_dotenv()

KEY = os.environ.get("GEMINI_API_KEY")          # ← 不要用 os.environ[...]
if not KEY:
    raise RuntimeError("找不到 GEMINI_API_KEY：本機放 .env，雲端放 App settings → Secrets")
client = genai.Client(api_key=KEY)
MODEL = "gemini-embedding-001"
DIM = 768        # 預設 3072；截短成 768 省記憶體與比對時間（靠 [[MRL]]）

# Gemini 的兩個限制，兩個都是 FAQ 時代（5 筆）不會碰到、換成職缺（140 塊）才炸出來的：
#   1. BatchEmbedContentsRequest 一次最多 100 筆 → 400 INVALID_ARGUMENT
#   2. 免費方案「每分鐘 100 個 request」，而 batch 裡每段文字各算一個 → 429
BATCH = 100
CACHE = Path(__file__).resolve().parent.parent / "data" / "vecs_cloud.npz"


def embed(texts, task_type):
    vecs = []
    for i in range(0, len(texts), BATCH):
        part = texts[i:i + BATCH]
        for attempt in range(6):
            try:
                resp = client.models.embed_content(
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


# 區名詞彙表：問句裡的「內湖」沒有「區」字可當錨點，只能靠詞彙表比對。
# 從 DOCS 長出來，所以重建 jobs.json 之後會自動跟著更新。
DISTRICTS = known_districts(DOCS)

_DOC_VECS = None            # import 時不打 API，第一次檢索才建索引


def doc_vecs():
    """建索引，並存成快取。

    地端重算只是慢十秒，雲端重算是燒 140 個 request 配額外加兩分鐘等待
    （免費方案每分鐘只給 100 個），所以雲端這側非快取不可。
    快取用 DOCS 的 text 算雜湊當 key——重建 jobs.json 之後會自動失效重算。
    """
    global _DOC_VECS
    if _DOC_VECS is not None:
        return _DOC_VECS

    texts = [d.text for d in DOCS]
    key = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        if str(z["key"]) == key:
            _DOC_VECS = z["vecs"]
            return _DOC_VECS

    _DOC_VECS = embed(texts, "RETRIEVAL_DOCUMENT")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, key=np.array(key), vecs=_DOC_VECS)
    return _DOC_VECS


def retrieve(question, k=5, min_score=0.65, filters=None):
    # 0.65 是對職缺資料量出來的（tools/probe_threshold.py cloud）。
    # 原本的 0.70 是 FAQ 時代的值，換成職缺後會擋掉「有沒有可以全遠端的
    # 工作？」——它只有 0.695。Gemini 的間隙有 0.092 寬（地端 e5 只有
    # 0.003），所以雲端這側門檻還真的擋得住離題，不像地端得交給模型判斷。
    """回傳 k 個「不同職缺」，不是 k 個塊。細節見 onperm/rag.py 的同名函式——
    兩邊形狀刻意保持一致，換邊只要換 sys.path 那一行。"""
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
    question = "有哪些無人機相關的職缺？"
    for doc, score in retrieve(question):
        print(f"{score:.4f}  {doc.label}")
